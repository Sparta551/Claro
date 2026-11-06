#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLARO NEXUS - Orquestador Next Best Action configurable
=======================================================

Version Python del motor integrado en CLARO_NEXUS_Motor_NBA.ipynb.

Incluye lectura de configuracion, interprete de reglas, elegibilidad, canales,
paquetes, asignacion global con OR-Tools (o heuristica), parrilla, validador
R01-R16, metricas y una consola web local para editar y ejecutar.

Uso (por defecto trabaja con la base completa de 100.000 clientes):
    python claro_nexus.py                      # = run --dataset full --out parrilla_100k.csv
    python claro_nexus.py run --dataset demo   # opcional: demo de 5 clientes
    python claro_nexus.py serve                # consola web (base completa)
    python claro_nexus.py serve --open
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
import time
import traceback
import webbrowser
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd

try:
    from ortools.graph.python import min_cost_flow

    ORTOOLS_OK = True
except Exception:
    ORTOOLS_OK = False

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_FECHA = "2026-11-04"
DEFAULT_DATASET = "full"
DEFAULT_OUT = "parrilla_100k.csv"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

CAMPANAS_COLUMNS = [
    "id", "nombre", "producto", "objetivo", "columna_score", "valor_negocio",
    "canales_permitidos", "vigencia_inicio", "vigencia_fin", "activa",
]
REGLAS_COLUMNS = [
    "id", "nombre", "tipo", "alcance_nivel", "alcance_valores", "atributo",
    "operador", "valor", "accion", "activa",
]
PARAMETROS_COLUMNS = ["parametro", "valor"]

DATASETS = {
    "demo": ("casos_ejemplo_clientes.csv", "Demostracion (5 clientes)"),
    "full": ("CLARO_NEXUS_parrilla_100k.csv", "Base completa (100.000 clientes)"),
    "synthetic": ("CLARO_NEXUS_clientes_sinteticos_5000.csv", "Sintetica (5.000 clientes)"),
}
OPERADORES = {"==", "!=", ">", ">=", "<", "<=", "en", "not_en", "contiene", "no_contiene"}
DATA_CACHE: Dict[str, pd.DataFrame] = {}

# ---------------------------------------------------------------------------
# Utilidades de configuracion y datos
# ---------------------------------------------------------------------------
def localizar_config_dir(config_dir: Optional[str] = None) -> Path:
    """Ubica la carpeta que contiene config_campanas/reglas/parametros."""
    if config_dir:
        p = Path(config_dir).expanduser().resolve()
        if not _es_config_dir(p):
            raise FileNotFoundError(f"No encontre la configuracion en: {p}")
        return p

    candidatos: List[Path] = [Path.cwd(), APP_DIR]
    try:
        candidatos += list(Path.home().joinpath("Downloads").glob("*Orquestador*"))
        candidatos += list(Path.home().joinpath("Downloads").glob("*Orquestador*/*"))
    except Exception:
        pass

    vistos = set()
    for p in candidatos:
        try:
            p = p.expanduser().resolve()
        except Exception:
            continue
        if p in vistos:
            continue
        vistos.add(p)
        if _es_config_dir(p):
            return p

    raise FileNotFoundError(
        "No encontre config_campanas.csv, config_reglas.csv y config_parametros.csv. "
        "Use --config-dir o ejecute el script desde la carpeta del proyecto."
    )


def _es_config_dir(p: Path) -> bool:
    return (
        (p / "config_campanas.csv").exists()
        and (p / "config_reglas.csv").exists()
        and (p / "config_parametros.csv").exists()
    )


def _to_float(valor: Any, campo: str) -> float:
    texto = str(valor).strip().replace(",", ".")
    try:
        return float(texto)
    except Exception as exc:
        raise ValueError(f"Valor numerico invalido en {campo}: {valor!r}") from exc


def _to_int01(valor: Any, campo: str = "activa") -> int:
    if isinstance(valor, bool):
        return int(valor)
    texto = str(valor).strip().lower()
    if texto in {"1", "1.0", "true", "si", "sí", "yes"}:
        return 1
    if texto in {"0", "0.0", "false", "no", ""}:
        return 0
    try:
        return 1 if float(texto) != 0 else 0
    except Exception as exc:
        raise ValueError(f"Valor 0/1 invalido en {campo}: {valor!r}") from exc


def _normalizar_campanas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CAMPANAS_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[CAMPANAS_COLUMNS].fillna("")
    for col in CAMPANAS_COLUMNS:
        if col not in {"valor_negocio", "activa"}:
            df[col] = df[col].astype(str).str.strip()
    df = df[df["id"] != ""].copy()
    if df.empty:
        raise ValueError("Debe existir al menos una campana con id.")
    if df["id"].duplicated().any():
        dup = sorted(df.loc[df["id"].duplicated(), "id"].unique())
        raise ValueError(f"Campanas con id duplicado: {dup}")
    df["valor_negocio"] = [_to_float(v, "valor_negocio") for v in df["valor_negocio"]]
    df["activa"] = [_to_int01(v) for v in df["activa"]]
    for col in ("vigencia_inicio", "vigencia_fin"):
        fechas = pd.to_datetime(df[col], errors="coerce")
        if fechas.isna().any():
            malas = df.loc[fechas.isna(), "id"].tolist()
            raise ValueError(f"Fechas invalidas en {col} para: {malas}")
        df[col] = fechas.dt.strftime("%Y-%m-%d")
    for col in ("producto", "columna_score", "canales_permitidos"):
        if (df[col] == "").any():
            raise ValueError(f"La columna {col} no puede estar vacia en campanas.")
    return df.reset_index(drop=True)


def _normalizar_reglas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in REGLAS_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[REGLAS_COLUMNS].fillna("").astype(str)
    df = df[df["id"].str.strip() != ""].copy()
    if df.empty:
        raise ValueError("Debe existir al menos una regla con id.")
    if df["id"].duplicated().any():
        dup = sorted(df.loc[df["id"].duplicated(), "id"].unique())
        raise ValueError(f"Reglas con id duplicado: {dup}")
    for col in ("id", "nombre", "tipo", "alcance_nivel", "alcance_valores", "atributo", "operador", "valor", "accion"):
        df[col] = df[col].str.strip()
    df["tipo"] = df["tipo"].str.lower()
    df["alcance_nivel"] = df["alcance_nivel"].str.lower()
    df["operador"] = df["operador"].str.lower()
    df["accion"] = df["accion"].str.lower()
    if not df["tipo"].isin({"elegibilidad", "canal"}).all():
        raise ValueError("tipo de regla debe ser 'elegibilidad' o 'canal'.")
    if not df["alcance_nivel"].isin({"global", "campana", "canal"}).all():
        raise ValueError("alcance_nivel debe ser global, campana o canal.")
    if not df["operador"].isin(OPERADORES).all():
        malos = sorted(set(df["operador"]) - OPERADORES)
        raise ValueError(f"Operadores no soportados: {malos}")
    if not (df["accion"] == "excluir").all():
        raise ValueError("La unica accion soportada es 'excluir'.")
    if (df["atributo"] == "").any():
        raise ValueError("El atributo de una regla no puede estar vacio.")
    df["activa"] = [_to_int01(v) for v in df["activa"]]
    return df.reset_index(drop=True)


def _normalizar_parametros(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in PARAMETROS_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[PARAMETROS_COLUMNS].fillna("").astype(str)
    df["parametro"] = df["parametro"].str.strip()
    df["valor"] = df["valor"].str.strip()
    df = df[df["parametro"] != ""].copy()
    if df.empty:
        raise ValueError("Debe existir al menos un parametro.")
    if df["parametro"].duplicated().any():
        dup = sorted(df.loc[df["parametro"].duplicated(), "parametro"].unique())
        raise ValueError(f"Parametros duplicados: {dup}")
    return df.reset_index(drop=True)


def cargar_configuracion(config_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    campanas = pd.read_csv(config_dir / "config_campanas.csv", dtype=str, keep_default_na=False)
    reglas = pd.read_csv(config_dir / "config_reglas.csv", dtype=str, keep_default_na=False)
    parametros = pd.read_csv(config_dir / "config_parametros.csv", dtype=str, keep_default_na=False)
    return (
        _normalizar_campanas(campanas),
        _normalizar_reglas(reglas),
        _normalizar_parametros(parametros),
    )


def guardar_configuracion(
    salida: Path,
    campanas: pd.DataFrame,
    reglas: pd.DataFrame,
    parametros: pd.DataFrame,
) -> Path:
    salida = Path(salida).resolve()
    salida.mkdir(parents=True, exist_ok=True)
    _normalizar_campanas(campanas).to_csv(salida / "config_campanas.csv", index=False, encoding="utf-8-sig")
    _normalizar_reglas(reglas).to_csv(salida / "config_reglas.csv", index=False, encoding="utf-8-sig")
    _normalizar_parametros(parametros).to_csv(salida / "config_parametros.csv", index=False, encoding="utf-8-sig")
    return salida


def cargar_datos(data_dir: Path, dataset: str = DEFAULT_DATASET) -> pd.DataFrame:
    data_dir = Path(data_dir).resolve()
    if dataset in DATASETS:
        nombre = DATASETS[dataset][0]
        ruta = data_dir / nombre
    else:
        ruta = Path(dataset).expanduser().resolve()
    if not ruta.exists():
        raise FileNotFoundError(f"No encontre el archivo de clientes: {ruta}")

    clave = f"{ruta}|{ruta.stat().st_mtime_ns}"
    if clave not in DATA_CACHE:
        df = pd.read_csv(ruta)
        if "id_cliente" not in df.columns:
            raise ValueError("El archivo de clientes no tiene la columna id_cliente.")
        if df["id_cliente"].duplicated().any():
            raise ValueError("El archivo de clientes tiene id_cliente duplicados.")
        DATA_CACHE.clear()
        DATA_CACHE[clave] = df
    return DATA_CACHE[clave]


def dicc_parametros(parametros: pd.DataFrame) -> Dict[str, str]:
    return parametros.set_index("parametro")["valor"].to_dict()


def param(cfg: Dict[str, str], nombre: str, defecto: Any = None, tipo: type = float) -> Any:
    if nombre not in cfg or pd.isna(cfg[nombre]) or str(cfg[nombre]).strip() == "":
        if defecto is not None:
            return defecto
        raise KeyError(f"Parametro no configurado: {nombre}")
    texto = str(cfg[nombre]).strip()
    if tipo is int:
        return int(float(texto.replace(",", ".")))
    if tipo is float:
        return float(texto.replace(",", "."))
    return tipo(texto)


def lista_param(cfg: Dict[str, str], nombre: str, defecto: str = "") -> List[str]:
    valor = cfg.get(nombre, defecto)
    return [x.strip() for x in str(valor).split("|") if x.strip()]


# ---------------------------------------------------------------------------
# Interprete de reglas y elegibilidad
# ---------------------------------------------------------------------------
def _condicion_escalar(x: Any, op: str, v: Any) -> bool:
    if op in ("==", "!="):
        try:
            resultado = float(x) == float(v)
        except (TypeError, ValueError):
            resultado = str(x).strip() == str(v).strip()
        return resultado if op == "==" else not resultado
    if op in (">", ">=", "<", "<="):
        try:
            x, v = float(x), float(v)
        except (TypeError, ValueError):
            return False
        return {">": x > v, ">=": x >= v, "<": x < v, "<=": x <= v}[op]
    if op in ("en", "not_en"):
        opciones = [p.strip() for p in str(v).split("|")]
        resultado = str(x).strip() in opciones
        return resultado if op == "en" else not resultado
    if op in ("contiene", "no_contiene"):
        buscado = str(v).strip()
        resultado = buscado in [p.strip() for p in str(x).split("|")]
        return resultado if op == "contiene" else not resultado
    raise ValueError(f"Operador no soportado: {op}")


def evaluar_condicion(serie: pd.Series, operador: str, valor: Any) -> pd.Series:
    op = str(operador).strip()
    if isinstance(valor, pd.Series):
        valor = valor.reindex(serie.index)
        return pd.Series(
            [_condicion_escalar(x, op, v) for x, v in zip(serie, valor)],
            index=serie.index,
        )
    if op in ("==", "!="):
        numerico = pd.to_numeric(serie, errors="coerce")
        valor_num = pd.to_numeric(pd.Series([valor]), errors="coerce").iloc[0]
        if pd.notna(valor_num) and numerico.notna().all():
            resultado = numerico == valor_num
        else:
            resultado = serie.astype(str).str.strip() == str(valor).strip()
        return resultado if op == "==" else ~resultado
    if op in (">", ">=", "<", "<="):
        num = pd.to_numeric(serie, errors="coerce")
        v = float(valor)
        return {">": num > v, ">=": num >= v, "<": num < v, "<=": num <= v}[op]
    if op in ("en", "not_en"):
        opciones = [x.strip() for x in str(valor).split("|")]
        resultado = serie.astype(str).str.strip().isin(opciones)
        return resultado if op == "en" else ~resultado
    if op in ("contiene", "no_contiene"):
        buscado = str(valor).strip()
        resultado = serie.fillna("").astype(str).apply(
            lambda texto: buscado in [p.strip() for p in texto.split("|")]
        )
        return resultado if op == "contiene" else ~resultado
    raise ValueError(f"Operador no soportado: {operador}")


def _reglas_por_tipo(reglas: pd.DataFrame, tipo: str) -> pd.DataFrame:
    return reglas.loc[(reglas["tipo"] == tipo) & (reglas["activa"].astype(int) == 1)]


def evaluar_elegibilidad(
    datos: pd.DataFrame, campanas: pd.DataFrame, reglas: pd.DataFrame
) -> pd.DataFrame:
    reglas_eleg = _reglas_por_tipo(reglas, "elegibilidad")
    resultados = []
    for _, campana in campanas.iterrows():
        cid = campana["id"]
        producto = campana["producto"]
        motivos = pd.Series("", index=datos.index, dtype=object)
        for _, regla in reglas_eleg.iterrows():
            nivel = str(regla["alcance_nivel"])
            if nivel == "campana":
                alcance = [x.strip() for x in str(regla["alcance_valores"]).split("|")]
                if cid not in alcance:
                    continue
            elif nivel != "global":
                continue
            atributo = str(regla["atributo"])
            if atributo not in datos.columns:
                raise ValueError(f"Columna de regla inexistente: {atributo}")
            valor = "$producto" if str(regla["valor"]) == "$producto" else str(regla["valor"])
            if valor == "$producto":
                valor = producto
            condicion = evaluar_condicion(datos[atributo], regla["operador"], valor).fillna(False)
            if str(regla["accion"]) == "excluir":
                motivos = motivos.mask(condicion, motivos + str(regla["id"]) + "|")
        tabla = pd.DataFrame({
            "id_cliente": datos["id_cliente"].to_numpy(),
            "campana": cid,
            "producto": producto,
            "reglas_exclusion": motivos.str.rstrip("|").to_numpy(),
        })
        tabla["apto_reglas"] = tabla["reglas_exclusion"] == ""
        resultados.append(tabla)
    if not resultados:
        return pd.DataFrame(columns=["id_cliente", "campana", "producto", "reglas_exclusion", "apto_reglas"])
    return pd.concat(resultados, ignore_index=True)


def diagnosticar_sin_campana(
    datos: pd.DataFrame,
    campanas: pd.DataFrame,
    elegibilidad: pd.DataFrame,
    umbral_producto: float,
    max_contactos: int,
) -> pd.Series:
    motivos = pd.DataFrame(index=datos.index)
    for _, campana in campanas.iterrows():
        cid = campana["id"]
        sub = elegibilidad.loc[elegibilidad["campana"] == cid].reset_index(drop=True)
        if len(sub) != len(datos):
            motivos[cid] = ""
            continue
        score = pd.to_numeric(datos[campana["columna_score"]], errors="coerce")
        motivo = sub["reglas_exclusion"].fillna("").to_numpy().astype(str).copy()
        mal_score = (score < umbral_producto).fillna(True).to_numpy()
        motivo = np.array(
            [
                ("score" if not m else f"{m}|score") if mal else m
                for m, mal in zip(motivo, mal_score)
            ],
            dtype=object,
        )
        motivos[cid] = np.where(motivo == "", "", [f"{cid}:{m}" for m in motivo])
    base = pd.Series(
        [";".join([x for x in fila if x]) for fila in motivos.to_numpy(dtype=object)],
        index=datos.index,
        dtype=object,
    )
    base = base.mask(
        pd.to_numeric(datos["contactos_semana_actual"], errors="coerce") >= max_contactos,
        "Tope semanal de contactos",
    )
    base.index = datos["id_cliente"].to_numpy()
    return base


# ---------------------------------------------------------------------------
# Alternativas y paquetes comerciales
# ---------------------------------------------------------------------------
def construir_paquetes(
    datos: pd.DataFrame,
    campanas: pd.DataFrame,
    reglas: pd.DataFrame,
    parametros: pd.DataFrame,
    fecha: str = DEFAULT_FECHA,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    cfg = dicc_parametros(parametros)
    fecha_eval = pd.Timestamp(fecha)

    activa = pd.to_numeric(campanas["activa"], errors="coerce").fillna(0).astype(int) == 1
    inicio = pd.to_datetime(campanas["vigencia_inicio"], errors="coerce")
    fin = pd.to_datetime(campanas["vigencia_fin"], errors="coerce")
    campanas_activas = campanas.loc[activa & (inicio <= fecha_eval) & (fin >= fecha_eval)].copy()

    faltantes_score = [c for c in campanas_activas["columna_score"] if c and c not in datos.columns]
    if faltantes_score:
        raise ValueError(f"Columnas de score de producto inexistentes: {faltantes_score}")

    elegibilidad = evaluar_elegibilidad(datos, campanas_activas, reglas)
    umbral_producto = param(cfg, "priorizacion.umbral_score_producto")
    umbral_canal = param(cfg, "canal.umbral_score_canal")
    max_contactos = int(param(cfg, "politica_contacto.max_contactos_semana"))
    diagnostico = diagnosticar_sin_campana(
        datos, campanas_activas, elegibilidad, umbral_producto, max_contactos
    )

    elegibilidad = elegibilidad.merge(
        campanas_activas[["id", "columna_score", "valor_negocio", "canales_permitidos"]].rename(
            columns={"id": "campana"}
        ),
        on="campana",
        how="inner",
        validate="many_to_one",
    )

    columnas_score = [c for c in campanas_activas["columna_score"].dropna().unique().tolist() if c]
    if columnas_score:
        scores = datos[["id_cliente"] + columnas_score].melt(
            id_vars="id_cliente", var_name="columna_score", value_name="score_producto"
        )
        elegibilidad = elegibilidad.merge(
            scores, on=["id_cliente", "columna_score"], how="left", validate="many_to_one"
        )
    else:
        elegibilidad["score_producto"] = np.nan

    elegibilidad = elegibilidad.merge(
        datos[["id_cliente", "contactos_semana_actual"]],
        on="id_cliente",
        how="left",
        validate="many_to_one",
    )

    elegibles = elegibilidad.loc[
        elegibilidad["apto_reglas"]
        & (pd.to_numeric(elegibilidad["score_producto"], errors="coerce") >= umbral_producto)
        & (pd.to_numeric(elegibilidad["contactos_semana_actual"], errors="coerce") < max_contactos)
    ].copy()

    columnas_paquete = [
        "id_cliente", "canal", "productos", "campanas", "cantidad_productos", "valor_total",
        "score_prioridad", "score_canal", "jerarquia_rank",
    ]
    if elegibles.empty:
        return pd.DataFrame(columns=columnas_paquete), elegibilidad, diagnostico

    elegibles["canal"] = elegibles["canales_permitidos"].str.split("|")
    opciones = elegibles.explode("canal").copy()
    opciones["canal"] = opciones["canal"].str.strip()

    columnas_canal = [c for c in datos.columns if c.startswith("score_canal_")]
    if not columnas_canal:
        raise ValueError("No hay columnas score_canal_<canal> en la base de clientes.")
    scores_canal = datos[["id_cliente"] + columnas_canal].melt(
        id_vars="id_cliente", var_name="columna_canal", value_name="score_canal"
    )
    scores_canal["canal"] = scores_canal["columna_canal"].str.replace("score_canal_", "", regex=False)
    opciones = opciones.merge(
        scores_canal[["id_cliente", "canal", "score_canal"]],
        on=["id_cliente", "canal"],
        how="inner",
        validate="many_to_one",
    )
    opciones = opciones.loc[
        pd.to_numeric(opciones["score_canal"], errors="coerce") >= umbral_canal
    ].copy()

    reglas_canal = _reglas_por_tipo(reglas, "canal")
    atributos_canal = sorted({str(a) for a in reglas_canal["atributo"]} - set(opciones.columns))
    if atributos_canal:
        opciones = opciones.merge(
            datos[["id_cliente"] + atributos_canal].drop_duplicates("id_cliente"),
            on="id_cliente",
            how="left",
            validate="many_to_one",
        )
    for _, regla in reglas_canal.iterrows():
        if str(regla["accion"]) != "excluir":
            raise ValueError(f"Accion de canal no soportada: {regla['accion']}")
        canales_regla = [x.strip() for x in str(regla["alcance_valores"]).split("|")]
        mask = opciones["canal"].isin(canales_regla)
        if not mask.any():
            continue
        atributo = str(regla["atributo"])
        if atributo not in opciones.columns:
            raise ValueError(f"Columna de canal inexistente: {atributo}")
        condicion = evaluar_condicion(
            opciones.loc[mask, atributo], regla["operador"], regla["valor"]
        ).fillna(False)
        opciones = opciones.drop(opciones.loc[mask].index[condicion])

    opciones["valor_esperado"] = (
        pd.to_numeric(opciones["score_producto"], errors="coerce")
        * pd.to_numeric(opciones["score_canal"], errors="coerce")
        * pd.to_numeric(opciones["valor_negocio"], errors="coerce")
    )

    jerarquia = lista_param(cfg, "priorizacion.jerarquia")
    if not jerarquia:
        jerarquia = list(campanas_activas["id"])
    rank = {cid: i for i, cid in enumerate(jerarquia)}
    n_jer = max(len(jerarquia) - 1, 1)
    opciones["jerarquia_rank"] = opciones["campana"].map(rank).fillna(len(jerarquia)).astype(int)
    opciones["nivel_jerarquia"] = np.where(
        opciones["jerarquia_rank"] < len(jerarquia),
        1 - opciones["jerarquia_rank"] / n_jer,
        0.0,
    )

    modo = str(cfg.get("priorizacion.modo", "ponderado")).strip()
    if modo == "jerarquia_estricta":
        opciones["score_prioridad"] = (
            -opciones["jerarquia_rank"] + 1e-6 * opciones["score_producto"].fillna(0)
        )
    else:
        w_score = param(cfg, "priorizacion.pesos.score_producto", 0.6)
        w_valor = param(cfg, "priorizacion.pesos.valor_negocio", 0.2)
        w_jer = param(cfg, "priorizacion.pesos.jerarquia", 0.2)
        opciones["score_prioridad"] = (
            w_score * opciones["score_producto"].fillna(0)
            + w_valor * opciones["valor_negocio"].fillna(0)
            + w_jer * opciones["nivel_jerarquia"]
        )

    limite_pref = "politica_contacto.max_productos_por_canal."
    limites = {
        str(f["parametro"]).replace(limite_pref, ""): int(float(f["valor"]))
        for _, f in parametros.iterrows()
        if str(f["parametro"]).startswith(limite_pref)
    }
    opciones["limite_productos"] = opciones["canal"].map(limites)
    if opciones["limite_productos"].isna().any():
        faltantes = sorted(opciones.loc[opciones["limite_productos"].isna(), "canal"].unique())
        raise ValueError(f"Canales sin tope de productos configurado: {faltantes}")

    desempate = lista_param(cfg, "priorizacion.desempate", "score_producto|valor_negocio")
    columnas_orden = ["id_cliente", "canal", "score_prioridad"]
    asc = [True, True, False]
    for col in desempate:
        if col in opciones.columns:
            columnas_orden.append(col)
            asc.append(False)
    opciones = opciones.sort_values(columnas_orden, ascending=asc)
    opciones["posicion"] = opciones.groupby(["id_cliente", "canal"]).cumcount() + 1
    mejores = opciones.loc[opciones["posicion"] <= opciones["limite_productos"]].copy()

    paquetes = mejores.groupby(["id_cliente", "canal"], as_index=False).agg(
        productos=("producto", "|".join),
        campanas=("campana", "|".join),
        cantidad_productos=("producto", "size"),
        valor_total=("valor_esperado", "sum"),
        score_prioridad=("score_prioridad", "max"),
        score_canal=("score_canal", "max"),
        jerarquia_rank=("jerarquia_rank", "min"),
    )
    return paquetes, elegibilidad, diagnostico


# ---------------------------------------------------------------------------
# Asignacion global
# ---------------------------------------------------------------------------
def leer_cupos(parametros: pd.DataFrame) -> Dict[str, int]:
    pref = "canal.capacidad_por_corrida."
    return {
        str(f["parametro"]).replace(pref, ""): int(float(f["valor"]))
        for _, f in parametros.iterrows()
        if str(f["parametro"]).startswith(pref)
    }


def asignar_optimo(paquetes: pd.DataFrame, cupos: Dict[str, int]) -> pd.DataFrame:
    if paquetes.empty:
        return paquetes.copy()
    if not ORTOOLS_OK:
        return asignar_heuristico(paquetes, cupos, "score_canal")

    paquetes = paquetes.reset_index(drop=True)
    clientes = paquetes["id_cliente"].drop_duplicates().tolist()
    n = len(clientes)
    nodo_cliente = {c: i + 1 for i, c in enumerate(clientes)}
    canales = list(cupos)
    nodo_canal = {c: n + 1 + i for i, c in enumerate(canales)}
    fuente, sumidero = 0, n + len(canales) + 1

    modelo = min_cost_flow.SimpleMinCostFlow()
    for cliente in clientes:
        modelo.add_arc_with_capacity_and_unit_cost(fuente, nodo_cliente[cliente], 1, 0)
        modelo.add_arc_with_capacity_and_unit_cost(nodo_cliente[cliente], sumidero, 1, 0)
    for canal, capacidad in cupos.items():
        modelo.add_arc_with_capacity_and_unit_cost(nodo_canal[canal], sumidero, int(capacidad), 0)

    escala = 1_000_000
    arcos = {}
    for fila in paquetes.itertuples(index=True):
        costo = -int(round(float(fila.valor_total) * escala))
        arco = modelo.add_arc_with_capacity_and_unit_cost(
            nodo_cliente[fila.id_cliente], nodo_canal[fila.canal], 1, costo
        )
        arcos[arco] = fila.Index

    modelo.set_node_supply(fuente, n)
    modelo.set_node_supply(sumidero, -n)
    estado = modelo.solve()
    if estado != modelo.OPTIMAL:
        raise RuntimeError(f"No se obtuvo solucion optima (estado {estado})")

    indices = [idx for arco, idx in arcos.items() if modelo.flow(arco) == 1]
    return paquetes.loc[indices].copy()


def asignar_heuristico(
    paquetes: pd.DataFrame,
    cupos: Dict[str, int],
    criterio: str,
    orden_preferencia: Optional[List[str]] = None,
) -> pd.DataFrame:
    if paquetes.empty:
        return paquetes.copy()
    trabajo = paquetes.copy()
    if criterio == "orden_preferencia" and orden_preferencia:
        rank = {c: i for i, c in enumerate(orden_preferencia)}
        trabajo["_orden"] = trabajo["canal"].map(rank).fillna(len(rank))
        trabajo = trabajo.sort_values(
            ["_orden", "score_prioridad", "valor_total"], ascending=[True, False, False]
        )
    else:
        trabajo = trabajo.sort_values(
            ["score_prioridad", "valor_total"], ascending=[False, False]
        )
    usados = set()
    uso = {c: 0 for c in cupos}
    filas = []
    for idx, fila in trabajo.iterrows():
        if fila["id_cliente"] in usados:
            continue
        if uso.get(fila["canal"], 0) >= cupos.get(fila["canal"], 0):
            continue
        usados.add(fila["id_cliente"])
        uso[fila["canal"]] = uso.get(fila["canal"], 0) + 1
        filas.append(idx)
    return paquetes.loc[filas].copy()


def asignar(paquetes: pd.DataFrame, cupos: Dict[str, int], parametros: pd.DataFrame) -> pd.DataFrame:
    cfg = dicc_parametros(parametros)
    criterio = str(cfg.get("canal.criterio", "score_canal")).strip()
    if criterio == "orden_preferencia":
        orden = lista_param(cfg, "canal.orden_preferencia")
        return asignar_heuristico(paquetes, cupos, criterio, orden)
    return asignar_optimo(paquetes, cupos)


# ---------------------------------------------------------------------------
# Parrilla y razones
# ---------------------------------------------------------------------------
def construir_parrilla(
    datos: pd.DataFrame,
    paquetes: pd.DataFrame,
    diagnostico: pd.Series,
    parametros: pd.DataFrame,
) -> pd.DataFrame:
    cfg = dicc_parametros(parametros)
    id_col = datos[["id_cliente"]].copy()
    asignadas = paquetes.copy()
    if asignadas.empty:
        asignadas = pd.DataFrame(
            columns=[
                "id_cliente", "canal", "campanas", "productos",
                "cantidad_productos", "valor_total",
            ]
        )
    split = asignadas["campanas"].str.split("|", regex=False)
    asignadas["campana_1"] = split.str[0]
    asignadas["campana_2"] = split.str[1]
    asignadas["producto_1"] = asignadas["productos"].str.split("|", regex=False).str[0]
    asignadas["producto_2"] = asignadas["productos"].str.split("|", regex=False).str[1]

    parrilla = id_col.merge(
        asignadas[
            [
                "id_cliente", "canal", "campana_1", "campana_2", "producto_1",
                "producto_2", "cantidad_productos", "valor_total",
            ]
        ],
        on="id_cliente",
        how="left",
        validate="one_to_one",
    )
    parrilla["estado"] = np.where(parrilla["canal"].notna(), "asignado", "sin_accion")

    criterio = str(cfg.get("canal.criterio", "score_canal")).strip()
    etiqueta_criterio = (
        "mayor valor esperado" if criterio == "score_canal" else "orden de preferencia"
    )

    parrilla["razon"] = parrilla["id_cliente"].map(diagnostico)
    parrilla["razon"] = parrilla["razon"].fillna("Sin campana elegible o sin cupo disponible")
    asignado = parrilla["estado"] == "asignado"
    parrilla.loc[asignado, "razon"] = [
        (
            f"Canal {canal} elegido por {etiqueta_criterio} con cupo; "
            f"{int(n)} producto(s) segun tope del canal (valor esperado {float(v):.4f})"
        )
        for canal, n, v in zip(
            parrilla.loc[asignado, "canal"],
            parrilla.loc[asignado, "cantidad_productos"],
            parrilla.loc[asignado, "valor_total"],
        )
    ]
    con_paquete = set(paquetes["id_cliente"]) if not paquetes.empty else set()
    sin_asignar = parrilla["estado"] == "sin_accion"
    parrilla.loc[sin_asignar & parrilla["id_cliente"].isin(con_paquete), "razon"] = (
        "Sin cupo disponible en sus canales para esta corrida"
    )
    return parrilla[
        ["id_cliente", "estado", "canal", "campana_1", "campana_2", "razon"]
    ]


# ---------------------------------------------------------------------------
# Validador independiente (R01-R16)
# ---------------------------------------------------------------------------
def validar_parrilla(
    parrilla: pd.DataFrame,
    datos: pd.DataFrame,
    campanas: pd.DataFrame,
    reglas: pd.DataFrame,
    parametros: pd.DataFrame,
    fecha: str = DEFAULT_FECHA,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = dicc_parametros(parametros)
    alertas: List[Dict[str, Any]] = []

    def registrar(restriccion: str, mascara: Any, ids: Any, detalle: str) -> None:
        mascara = pd.Series(mascara).fillna(False).reset_index(drop=True).astype(bool)
        ids = pd.Series(ids).reset_index(drop=True)
        for cliente in ids.loc[mascara]:
            alertas.append(
                {"restriccion": restriccion, "id_cliente": cliente, "detalle": detalle}
            )

    requeridas = ["id_cliente", "estado", "canal", "campana_1", "campana_2", "razon"]
    faltan = [c for c in requeridas if c not in parrilla.columns]
    if faltan:
        raise ValueError(f"Faltan columnas de parrilla: {faltan}")

    p = parrilla.copy()
    registrar(
        "R13_UN_CANAL", p["id_cliente"].duplicated(keep=False), p["id_cliente"],
        "Cliente repetido en la parrilla",
    )
    registrar(
        "R13_UN_CANAL", ~p["estado"].isin(["asignado", "sin_accion"]), p["id_cliente"],
        "Estado invalido",
    )
    registrar(
        "R13_UN_CANAL", (p["estado"] == "asignado") & p["canal"].isna(), p["id_cliente"],
        "Asignado sin canal",
    )
    registrar(
        "R13_UN_CANAL",
        (p["estado"] == "sin_accion") & p[["canal", "campana_1", "campana_2"]].notna().any(axis=1),
        p["id_cliente"],
        "Sin accion con datos de contacto",
    )

    ofertas = p.loc[p["estado"] == "asignado"].melt(
        id_vars=["id_cliente", "canal"],
        value_vars=["campana_1", "campana_2"],
        value_name="campana",
    ).dropna(subset=["campana"])
    columnas_alerta = ["restriccion", "id_cliente", "detalle"]
    if ofertas.empty:
        return pd.DataFrame(alertas, columns=columnas_alerta), ofertas

    catalogo = campanas[
        [
            "id", "producto", "columna_score", "valor_negocio",
            "canales_permitidos", "activa", "vigencia_inicio", "vigencia_fin",
        ]
    ].rename(columns={"id": "campana"})
    ofertas = ofertas.merge(catalogo, on="campana", how="left", validate="many_to_one")
    ofertas = ofertas.merge(datos, on="id_cliente", how="left", validate="many_to_one")

    registrar(
        "CAMPANA_DESCONOCIDA", ofertas["producto"].isna(), ofertas["id_cliente"],
        "Campana inexistente",
    )
    fecha_eval = pd.Timestamp(fecha)
    vigente = (pd.to_datetime(ofertas["vigencia_inicio"], errors="coerce") <= fecha_eval) & (
        pd.to_datetime(ofertas["vigencia_fin"], errors="coerce") >= fecha_eval
    )
    registrar(
        "CAMPANA_INACTIVA",
        ofertas["activa"].notna() & (pd.to_numeric(ofertas["activa"], errors="coerce") != 1),
        ofertas["id_cliente"],
        "Campana inactiva",
    )
    registrar(
        "CAMPANA_VENCIDA", ofertas["producto"].notna() & ~vigente, ofertas["id_cliente"],
        "Campana fuera de vigencia",
    )

    # R01-R07 y reglas de elegibilidad globales / por campana
    for _, regla in _reglas_por_tipo(reglas, "elegibilidad").iterrows():
        nivel = str(regla["alcance_nivel"])
        if nivel == "global":
            mascara = pd.Series(True, index=ofertas.index)
        elif nivel == "campana":
            alcance = [x.strip() for x in str(regla["alcance_valores"]).split("|")]
            mascara = ofertas["campana"].isin(alcance)
        else:
            continue
        if not mascara.any():
            continue
        atributo = str(regla["atributo"])
        if atributo not in ofertas.columns:
            continue
        valor: Any = "$producto" if str(regla["valor"]) == "$producto" else str(regla["valor"])
        if str(valor) == "$producto":
            valor = ofertas["producto"]
        condicion = evaluar_condicion(
            ofertas[atributo], regla["operador"], valor
        ).fillna(False)
        if str(regla["accion"]) == "excluir":
            registrar(
                str(regla["id"]),
                mascara & condicion,
                ofertas["id_cliente"],
                f"Incumple {regla['id']}: {regla['nombre']}",
            )

    # R08 score minimo de producto
    umbral_producto = param(cfg, "priorizacion.umbral_score_producto")
    ofertas["score_producto"] = np.nan
    for col in campanas["columna_score"].dropna().unique():
        if col in ofertas.columns:
            m = ofertas["columna_score"] == col
            ofertas.loc[m, "score_producto"] = pd.to_numeric(
                ofertas.loc[m, col], errors="coerce"
            )
    registrar(
        "R08_SCORE_PRODUCTO",
        pd.to_numeric(ofertas["score_producto"], errors="coerce") < umbral_producto,
        ofertas["id_cliente"],
        "Score de producto bajo",
    )

    # R09 autorizacion de canales digitales y R10 limite de asesor
    for _, regla in _reglas_por_tipo(reglas, "canal").iterrows():
        canales = [x.strip() for x in str(regla["alcance_valores"]).split("|")]
        if str(regla["accion"]) != "excluir":
            continue
        atributo = str(regla["atributo"])
        if atributo not in ofertas.columns:
            continue
        mask = ofertas["canal"].isin(canales)
        if not mask.any():
            continue
        cond = evaluar_condicion(
            ofertas.loc[mask, atributo], regla["operador"], regla["valor"]
        ).fillna(False)
        violacion = pd.Series(False, index=ofertas.index)
        violacion.loc[cond.loc[cond].index] = True
        codigo = "R09_AUTORIZACION" if "autoriza" in atributo else "R10_ASESOR"
        registrar(
            codigo, violacion, ofertas["id_cliente"],
            f"Incumple {regla['id']}: {regla['nombre']}",
        )

    # R11 score minimo de canal
    umbral_canal = param(cfg, "canal.umbral_score_canal")
    ofertas["score_canal"] = np.nan
    for canal in ofertas["canal"].dropna().unique():
        col = f"score_canal_{canal}"
        if col in ofertas.columns:
            m = ofertas["canal"] == canal
            ofertas.loc[m, "score_canal"] = pd.to_numeric(
                ofertas.loc[m, col], errors="coerce"
            )
    registrar(
        "R11_SCORE_CANAL",
        pd.to_numeric(ofertas["score_canal"], errors="coerce") < umbral_canal,
        ofertas["id_cliente"],
        "Score de canal bajo",
    )

    # R12 canal permitido por campana
    permitido = [
        canal in [x.strip() for x in str(permitidos).split("|")]
        for canal, permitidos in zip(ofertas["canal"], ofertas["canales_permitidos"])
    ]
    registrar(
        "R12_CANAL_CAMPANA", [not x for x in permitido], ofertas["id_cliente"],
        "Canal no permitido por la campana",
    )

    # R14 tope de productos por contacto
    pref = "politica_contacto.max_productos_por_canal."
    limites = {
        str(f["parametro"]).replace(pref, ""): int(float(f["valor"]))
        for _, f in parametros.iterrows()
        if str(f["parametro"]).startswith(pref)
    }
    conteo = (
        ofertas.groupby(["id_cliente", "canal"]).size().rename("n_productos").reset_index()
    )
    conteo["limite"] = conteo["canal"].map(limites)
    registrar(
        "R14_TOPE_PRODUCTOS",
        (conteo["n_productos"] > conteo["limite"]).fillna(False),
        conteo["id_cliente"],
        "Excede productos por contacto",
    )

    # R15 tope semanal
    max_contactos = int(param(cfg, "politica_contacto.max_contactos_semana"))
    asignados = p.loc[p["estado"] == "asignado", ["id_cliente"]].merge(
        datos, on="id_cliente", how="left", validate="one_to_one"
    )
    registrar(
        "R15_TOPE_SEMANAL",
        pd.to_numeric(asignados["contactos_semana_actual"], errors="coerce") >= max_contactos,
        asignados["id_cliente"],
        "Supera contactos semanales",
    )

    # R16 cupo por canal (contactos, no productos)
    cupos = leer_cupos(parametros)
    uso = p.loc[p["estado"] == "asignado", "canal"].value_counts()
    for canal, capacidad in cupos.items():
        if int(uso.get(canal, 0)) > int(capacidad):
            registrar(
                "R16_CUPO_CANAL", [True], [f"Canal {canal}"],
                f"Uso {int(uso.get(canal, 0))} > cupo {capacidad}",
            )

    detalle = pd.DataFrame(alertas, columns=columnas_alerta)
    return detalle, ofertas


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def calcular_metricas(
    parrilla: pd.DataFrame,
    ofertas: pd.DataFrame,
    datos: pd.DataFrame,
    campanas: pd.DataFrame,
    parametros: pd.DataFrame,
) -> Dict[str, Any]:
    valor = 0.0
    productos = 0
    if not ofertas.empty:
        ofertas = ofertas.copy()
        ofertas["valor"] = (
            pd.to_numeric(ofertas["score_producto"], errors="coerce")
            * pd.to_numeric(ofertas["score_canal"], errors="coerce")
            * pd.to_numeric(ofertas["valor_negocio"], errors="coerce")
        )
        valor = float(ofertas["valor"].sum())
        productos = int(len(ofertas))
    asignados = int((parrilla["estado"] == "asignado").sum())
    return {
        "clientes": len(parrilla),
        "clientes_asignados": asignados,
        "cobertura_pct": round(100 * asignados / len(parrilla), 2) if len(parrilla) else 0,
        "productos_asignados": productos,
        "valor_esperado": round(valor, 6),
    }


def calcular_jerarquia(
    parrilla: pd.DataFrame,
    elegibilidad: pd.DataFrame,
    parametros: pd.DataFrame,
) -> Dict[str, Optional[float]]:
    cfg = dicc_parametros(parametros)
    jerarquia = lista_param(cfg, "priorizacion.jerarquia")
    if not jerarquia:
        return {"jerarquia_pct_elegibles": None, "jerarquia_pct_asignados": None}
    rank = {cid: i for i, cid in enumerate(jerarquia)}

    umbral_producto = param(cfg, "priorizacion.umbral_score_producto")
    max_contactos = int(param(cfg, "politica_contacto.max_contactos_semana"))
    eleg = elegibilidad.copy()
    eleg["score_producto"] = pd.to_numeric(eleg["score_producto"], errors="coerce")
    eleg["contactos"] = pd.to_numeric(eleg["contactos_semana_actual"], errors="coerce")
    eleg = eleg.loc[
        eleg["apto_reglas"]
        & (eleg["score_producto"] >= umbral_producto)
        & (eleg["contactos"] < max_contactos)
    ].copy()
    eleg["rank"] = eleg["campana"].map(rank)

    mejor_elegible = eleg.dropna(subset=["rank"]).groupby("id_cliente")["rank"].min()

    ofertas = parrilla.loc[parrilla["estado"] == "asignado"].melt(
        id_vars=["id_cliente"], value_vars=["campana_1", "campana_2"], value_name="campana"
    ).dropna(subset=["campana"])
    ofertas["rank"] = ofertas["campana"].map(rank)
    mejor_asignado = ofertas.dropna(subset=["rank"]).groupby("id_cliente")["rank"].min()

    comunes = mejor_elegible.index.intersection(mejor_asignado.index)
    pct_eleg: Optional[float] = None
    if len(mejor_elegible) > 0:
        cumple = (mejor_asignado.reindex(mejor_elegible.index) == mejor_elegible).sum()
        pct_eleg = round(100 * cumple / len(mejor_elegible), 2)
    pct_asig: Optional[float] = None
    if len(comunes) > 0:
        cumple = (mejor_asignado.loc[comunes] == mejor_elegible.loc[comunes]).sum()
        pct_asig = round(100 * cumple / len(comunes), 2)
    return {"jerarquia_pct_elegibles": pct_eleg, "jerarquia_pct_asignados": pct_asig}


def ejecutar_orquestador(
    datos: pd.DataFrame,
    campanas: pd.DataFrame,
    reglas: pd.DataFrame,
    parametros: pd.DataFrame,
    fecha: str = DEFAULT_FECHA,
) -> Dict[str, Any]:
    inicio = time.perf_counter()
    paquetes, elegibilidad, diagnostico = construir_paquetes(
        datos, campanas, reglas, parametros, fecha
    )
    cupos = leer_cupos(parametros)
    asignacion = asignar(paquetes, cupos, parametros)
    parrilla = construir_parrilla(datos, asignacion, diagnostico, parametros)
    detalle, ofertas = validar_parrilla(
        parrilla, datos, campanas, reglas, parametros, fecha
    )
    metricas = calcular_metricas(parrilla, ofertas, datos, campanas, parametros)
    metricas.update(calcular_jerarquia(parrilla, elegibilidad, parametros))
    metricas["violaciones"] = int(len(detalle))
    metricas["tiempo_s"] = round(time.perf_counter() - inicio, 2)
    metricas["motor"] = "ortools" if ORTOOLS_OK else "heuristica"
    return {
        "parrilla": parrilla,
        "asignacion": asignacion,
        "paquetes": paquetes,
        "elegibilidad": elegibilidad,
        "diagnostico": diagnostico,
        "validacion": detalle,
        "ofertas": ofertas,
        "cupos": cupos,
        "metricas": metricas,
    }


# ---------------------------------------------------------------------------
# Edicion de configuracion (consola)
# ---------------------------------------------------------------------------
def actualizar_parametros(
    parametros: pd.DataFrame, cambios: Dict[str, Any]
) -> pd.DataFrame:
    actualizados = parametros.copy(deep=True)
    for nombre, valor in cambios.items():
        mascara = actualizados["parametro"] == nombre
        if not mascara.any():
            raise KeyError(f"Parametro inexistente: {nombre}")
        actualizados.loc[mascara, "valor"] = str(valor)
    return actualizados


def actualizar_regla(
    reglas: pd.DataFrame, id_regla: str, cambios: Dict[str, Any]
) -> pd.DataFrame:
    actualizadas = reglas.copy(deep=True)
    mascara = actualizadas["id"] == id_regla
    if not mascara.any():
        raise KeyError(f"Regla inexistente: {id_regla}")
    _aplicar_cambios(actualizadas, mascara, cambios, "regla")
    return actualizadas


def actualizar_campana(
    campanas: pd.DataFrame, id_campana: str, cambios: Dict[str, Any]
) -> pd.DataFrame:
    actualizadas = campanas.copy(deep=True)
    mascara = actualizadas["id"] == id_campana
    if not mascara.any():
        raise KeyError(f"Campana inexistente: {id_campana}")
    _aplicar_cambios(actualizadas, mascara, cambios, "campana")
    return actualizadas


def _aplicar_cambios(
    df: pd.DataFrame, mascara: pd.Series, cambios: Dict[str, Any], entidad: str
) -> None:
    for campo, valor in cambios.items():
        if campo not in df.columns:
            raise KeyError(f"Campo de {entidad} inexistente: {campo}")
        df.loc[mascara, campo] = valor


def agregar_regla(reglas: pd.DataFrame, datos: Dict[str, Any]) -> pd.DataFrame:
    nueva = {col: str(datos.get(col, "")).strip() for col in REGLAS_COLUMNS}
    if not nueva["id"]:
        raise ValueError("La nueva regla necesita un id.")
    if (reglas["id"] == nueva["id"]).any():
        raise ValueError(f"Ya existe una regla con id {nueva['id']}.")
    return pd.concat([reglas, pd.DataFrame([nueva])], ignore_index=True)


def eliminar_regla(reglas: pd.DataFrame, id_regla: str) -> pd.DataFrame:
    if (reglas["id"] == id_regla).sum() == 0:
        raise KeyError(f"Regla inexistente: {id_regla}")
    return reglas.loc[reglas["id"] != id_regla].reset_index(drop=True)


def agregar_campana(campanas: pd.DataFrame, datos: Dict[str, Any]) -> pd.DataFrame:
    nueva = {col: str(datos.get(col, "")).strip() for col in CAMPANAS_COLUMNS}
    if not nueva["id"]:
        raise ValueError("La nueva campana necesita un id.")
    if (campanas["id"] == nueva["id"]).any():
        raise ValueError(f"Ya existe una campana con id {nueva['id']}.")
    return pd.concat([campanas, pd.DataFrame([nueva])], ignore_index=True)


def eliminar_campana(campanas: pd.DataFrame, id_campana: str) -> pd.DataFrame:
    if (campanas["id"] == id_campana).sum() == 0:
        raise KeyError(f"Campana inexistente: {id_campana}")
    return campanas.loc[campanas["id"] != id_campana].reset_index(drop=True)


def validar_configuracion(
    campanas: pd.DataFrame,
    reglas: pd.DataFrame,
    parametros: pd.DataFrame,
) -> List[str]:
    """Avisos no bloqueantes sobre coherencia de la configuracion."""
    avisos: List[str] = []
    try:
        campanas = _normalizar_campanas(campanas)
        reglas = _normalizar_reglas(reglas)
        parametros = _normalizar_parametros(parametros)
    except Exception as exc:
        return [str(exc)]

    cfg = dicc_parametros(parametros)
    canales_campana = {
        c.strip()
        for valor in campanas["canales_permitidos"]
        for c in str(valor).split("|")
        if c.strip()
    }
    cupos = leer_cupos(parametros)
    sin_cupo = sorted(canales_campana - set(cupos))
    if sin_cupo:
        avisos.append(f"Canales sin capacidad configurada: {', '.join(sin_cupo)}")

    pref = "politica_contacto.max_productos_por_canal."
    canales_tope = {
        str(f["parametro"]).replace(pref, "")
        for _, f in parametros.iterrows()
        if str(f["parametro"]).startswith(pref)
    }
    sin_tope = sorted(canales_campana - canales_tope)
    if sin_tope:
        avisos.append(f"Canales sin tope de productos: {', '.join(sin_tope)}")

    jerarquia = lista_param(cfg, "priorizacion.jerarquia")
    ids = set(campanas["id"])
    huerfanas = [c for c in jerarquia if c not in ids]
    if huerfanas:
        avisos.append(f"Jerarquia incluye campanas inexistentes: {', '.join(huerfanas)}")
    sin_jerarquia = sorted(ids - set(jerarquia))
    if sin_jerarquia:
        avisos.append(
            "Campanas fuera de la jerarquia (quedan al final): "
            + ", ".join(sin_jerarquia)
        )

    for col in ("priorizacion.umbral_score_producto", "canal.umbral_score_canal"):
        try:
            valor = param(cfg, col)
            if not 0 <= valor <= 1:
                avisos.append(f"{col} fuera de [0,1]: {valor}")
        except Exception as exc:
            avisos.append(f"{col}: {exc}")

    pesos = [k for k in cfg if k.startswith("priorizacion.pesos.")]
    suma = 0.0
    for k in pesos:
        try:
            suma += float(cfg[k])
        except Exception:
            avisos.append(f"Peso no numerico: {k}={cfg[k]}")
    if pesos and abs(suma - 1.0) > 0.01:
        avisos.append(f"Los pesos de priorizacion suman {suma:.3f} (se espera 1.0)")

    modo = str(cfg.get("priorizacion.modo", "")).strip()
    if modo not in {"ponderado", "jerarquia_estricta"}:
        avisos.append(f"priorizacion.modo desconocido: {modo!r}")

    criterio = str(cfg.get("canal.criterio", "")).strip()
    if criterio not in {"score_canal", "orden_preferencia"}:
        avisos.append(f"canal.criterio desconocido: {criterio!r}")

    inactivas = campanas.loc[campanas["activa"] == 0, "id"].tolist()
    if inactivas:
        avisos.append(f"Campanas inactivas (no se usan): {', '.join(inactivas)}")

    vigencia_invalida = campanas.loc[
        pd.to_datetime(campanas["vigencia_inicio"]) > pd.to_datetime(campanas["vigencia_fin"]),
        "id",
    ].tolist()
    if vigencia_invalida:
        avisos.append(
            f"Campanas con vigencia_inicio > vigencia_fin: {', '.join(vigencia_invalida)}"
        )
    return avisos


# ---------------------------------------------------------------------------
# Serializacion de resultados
# ---------------------------------------------------------------------------
def _json_seguro(valor: Any) -> Any:
    if valor is None:
        return None
    if isinstance(valor, (bool, np.bool_)):
        return bool(valor)
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, (float, np.floating)):
        numero = float(valor)
        if math.isnan(numero) or math.isinf(numero):
            return None
        return numero
    if isinstance(valor, (pd.Timestamp, datetime, date)):
        return valor.strftime("%Y-%m-%d")
    try:
        if pd.isna(valor):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(valor, int):
        return valor
    return str(valor)


def df_registros(df: Optional[pd.DataFrame], limite: Optional[int] = None) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    if limite is not None:
        df = df.head(limite)
    return [
        {col: _json_seguro(valor) for col, valor in fila.items()}
        for fila in df.to_dict("records")
    ]


def resumen_ejecucion(
    resultado: Dict[str, Any],
    dataset: str,
    fecha: str,
    limite_muestra: int = 200,
    limite_violaciones: int = 200,
) -> Dict[str, Any]:
    parrilla = resultado["parrilla"]
    metricas = dict(resultado["metricas"])

    canales: List[Dict[str, Any]] = []
    asignacion = resultado["asignacion"]
    if not asignacion.empty:
        canal_df = (
            asignacion.groupby("canal", as_index=False)
            .agg(
                clientes=("id_cliente", "nunique"),
                productos=("cantidad_productos", "sum"),
                valor=("valor_total", "sum"),
            )
            .sort_values("valor", ascending=False)
        )
        cupos = resultado.get("cupos", {})
        canal_df["cupo"] = canal_df["canal"].map(cupos)
        canal_df["uso_pct"] = (
            canal_df["clientes"] / canal_df["cupo"].replace(0, np.nan) * 100
        )
        canales = df_registros(canal_df)

    ofertas = resultado.get("ofertas")
    campanas_resumen: List[Dict[str, Any]] = []
    productos_resumen: List[Dict[str, Any]] = []
    if ofertas is not None and not ofertas.empty:
        of = ofertas.copy()
        of["valor"] = (
            pd.to_numeric(of["score_producto"], errors="coerce")
            * pd.to_numeric(of["score_canal"], errors="coerce")
            * pd.to_numeric(of["valor_negocio"], errors="coerce")
        )
        camp = (
            of.groupby("campana", as_index=False)
            .agg(clientes=("id_cliente", "nunique"), productos=("campana", "size"), valor=("valor", "sum"))
            .sort_values("valor", ascending=False)
        )
        campanas_resumen = df_registros(camp)
        prod = (
            of.groupby("producto", as_index=False)
            .agg(clientes=("id_cliente", "nunique"), ofertas=("producto", "size"), valor=("valor", "sum"))
            .sort_values("valor", ascending=False)
        )
        productos_resumen = df_registros(prod)

    validacion = resultado["validacion"]
    if validacion.empty:
        violaciones_resumen: List[Dict[str, Any]] = []
    else:
        violaciones_resumen = df_registros(
            validacion.groupby("restriccion", as_index=False)
            .size()
            .rename(columns={"size": "casos"})
            .sort_values("casos", ascending=False)
        )

    sin_accion = parrilla.loc[parrilla["estado"] == "sin_accion", "razon"]
    razones: List[Dict[str, Any]] = []
    if not sin_accion.empty:
        razones = df_registros(
            sin_accion.value_counts().rename_axis("razon").reset_index(name="clientes")
        )

    return {
        "ok": True,
        "dataset": dataset,
        "fecha": fecha,
        "metricas": {k: _json_seguro(v) for k, v in metricas.items()},
        "canales": canales,
        "campanas": campanas_resumen,
        "productos": productos_resumen,
        "violaciones_resumen": violaciones_resumen,
        "violaciones": df_registros(validacion, limite_violaciones),
        "razones": razones,
        "muestra": df_registros(parrilla, limite_muestra),
        "total_filas": int(len(parrilla)),
        "avisos": validar_configuracion(
            resultado.get("_campanas"), resultado.get("_reglas"), resultado.get("_parametros")
        ) if resultado.get("_campanas") is not None else [],
    }


# ---------------------------------------------------------------------------
# Estado del servidor
# ---------------------------------------------------------------------------
STATE: Dict[str, Any] = {
    "config_dir": None,
    "campanas": None,
    "reglas": None,
    "parametros": None,
    "dataset": DEFAULT_DATASET,
    "fecha": DEFAULT_FECHA,
    "resultado": None,
    "resumen": None,
    "parrilla_csv": None,
    "historial": [],
}
STATE_LOCK = threading.RLock()


def _cargar_estado_inicial(config_dir: Optional[str], dataset: str, fecha: str) -> Path:
    ruta = localizar_config_dir(config_dir)
    campanas, reglas, parametros = cargar_configuracion(ruta)
    with STATE_LOCK:
        STATE["config_dir"] = ruta
        STATE["campanas"] = campanas
        STATE["reglas"] = reglas
        STATE["parametros"] = parametros
        STATE["dataset"] = dataset
        STATE["fecha"] = fecha
    return ruta


def _paquetes_config(body: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    campanas = body.get("campanas")
    reglas = body.get("reglas")
    parametros = body.get("parametros")
    if campanas is None or reglas is None or parametros is None:
        raise ValueError("Faltan campanas, reglas o parametros en la peticion.")
    df_camp = _normalizar_campanas(pd.DataFrame(campanas))
    df_reg = _normalizar_reglas(pd.DataFrame(reglas))
    df_par = _normalizar_parametros(pd.DataFrame(parametros))
    return df_camp, df_reg, df_par


def ejecutar_desde_consola(body: Dict[str, Any]) -> Dict[str, Any]:
    dataset = str(body.get("dataset") or STATE["dataset"] or DEFAULT_DATASET)
    fecha = str(body.get("fecha") or STATE["fecha"] or DEFAULT_FECHA)
    pd.Timestamp(fecha)

    if body.get("campanas") is not None:
        campanas, reglas, parametros = _paquetes_config(body)
    else:
        campanas = STATE["campanas"]
        reglas = STATE["reglas"]
        parametros = STATE["parametros"]
        if campanas is None:
            raise ValueError("No hay configuracion cargada.")

    avisos = validar_configuracion(campanas, reglas, parametros)
    config_dir = STATE["config_dir"] or localizar_config_dir()
    datos = cargar_datos(Path(config_dir), dataset)

    fecha_eval = pd.Timestamp(fecha)
    activas = campanas.loc[
        (pd.to_numeric(campanas["activa"], errors="coerce").fillna(0).astype(int) == 1)
        & (pd.to_datetime(campanas["vigencia_inicio"], errors="coerce") <= fecha_eval)
        & (pd.to_datetime(campanas["vigencia_fin"], errors="coerce") >= fecha_eval)
    ]
    faltantes = [c for c in activas["columna_score"] if c not in datos.columns]
    if faltantes:
        raise ValueError(
            "El dataset no tiene las columnas de score activas: "
            + ", ".join(sorted(set(faltantes)))
        )

    resultado = ejecutar_orquestador(datos, campanas, reglas, parametros, fecha)
    resultado["_campanas"] = campanas
    resultado["_reglas"] = reglas
    resultado["_parametros"] = parametros

    resumen = resumen_ejecucion(resultado, dataset, fecha)
    resumen["avisos"] = avisos
    resumen["datos"] = {
        "filas": int(len(datos)),
        "columnas": int(len(datos.columns)),
        "archivo": DATASETS.get(dataset, (dataset, dataset))[0],
    }
    resumen["motor"] = "ortools" if ORTOOLS_OK else "heuristica"
    resumen["hora"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    csv = resultado["parrilla"].to_csv(index=False).encode("utf-8-sig")
    with STATE_LOCK:
        STATE["campanas"] = campanas
        STATE["reglas"] = reglas
        STATE["parametros"] = parametros
        STATE["dataset"] = dataset
        STATE["fecha"] = fecha
        STATE["resultado"] = resultado
        STATE["resumen"] = resumen
        STATE["parrilla_csv"] = csv
        STATE["historial"].insert(
            0,
            {
                "hora": resumen["hora"],
                "dataset": dataset,
                "fecha": fecha,
                "metricas": resumen["metricas"],
                "motor": resumen["motor"],
            },
        )
        STATE["historial"] = STATE["historial"][:20]
    return resumen


# ---------------------------------------------------------------------------
# Servidor HTTP local
# ---------------------------------------------------------------------------
class NexusHTTPHandler(BaseHTTPRequestHandler):
    server_version = "ClaroNexus/1.0"

    def log_message(self, formato: str, *args: Any) -> None:
        sys.stderr.write(
            "[%s] %s\n" % (datetime.now().strftime("%H:%M:%S"), formato % args)
        )

    # -- helpers ------------------------------------------------------------
    def _enviar(self, cuerpo: bytes, content_type: str, codigo: int = 200, extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(codigo)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(cuerpo)

    def _json(self, datos: Any, codigo: int = 200) -> None:
        cuerpo = json.dumps(datos, ensure_ascii=False, default=str).encode("utf-8")
        self._enviar(cuerpo, "application/json; charset=utf-8", codigo)

    def _error(self, mensaje: str, codigo: int = 400, detalle: str = "") -> None:
        self._json(
            {"ok": False, "error": mensaje, "detalle": (detalle or "")[-1500:]},
            codigo,
        )

    def _leer_json(self) -> Dict[str, Any]:
        largo = int(self.headers.get("Content-Length") or 0)
        if largo <= 0:
            return {}
        crudo = self.rfile.read(largo)
        try:
            datos = json.loads(crudo.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"JSON invalido: {exc}") from exc
        if not isinstance(datos, dict):
            raise ValueError("El cuerpo debe ser un objeto JSON.")
        return datos

    # -- rutas --------------------------------------------------------------
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        ruta = urlparse(self.path).path
        if ruta in ("/", "/index.html", "/consola_nba.html"):
            return self._servir_consola()
        if ruta == "/api/health":
            return self._json(
                {
                    "ok": True,
                    "ortools": ORTOOLS_OK,
                    "python": sys.version.split()[0],
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                    "config_dir": str(STATE["config_dir"] or ""),
                    "dataset": STATE["dataset"],
                    "fecha": STATE["fecha"],
                }
            )
        if ruta == "/api/config":
            return self._servir_config()
        if ruta == "/api/datasets":
            return self._servir_datasets()
        if ruta == "/api/history":
            with STATE_LOCK:
                return self._json({"ok": True, "historial": STATE["historial"]})
        if ruta == "/api/parrilla.csv":
            return self._servir_parrilla()
        return self._error("Ruta no encontrada", 404)

    def do_POST(self) -> None:  # noqa: N802
        ruta = urlparse(self.path).path
        try:
            body = self._leer_json()
        except ValueError as exc:
            return self._error(str(exc), 400)

        if ruta == "/api/run":
            return self._manejar_run(body)
        if ruta == "/api/save":
            return self._manejar_save(body)
        if ruta == "/api/reset":
            return self._manejar_reset()
        if ruta == "/api/history/clear":
            with STATE_LOCK:
                STATE["historial"] = []
            return self._json({"ok": True})
        return self._error("Ruta no encontrada", 404)

    # -- implementaciones ---------------------------------------------------
    def _servir_consola(self) -> None:
        archivo = APP_DIR / "consola_nba.html"
        if not archivo.exists():
            return self._error(
                "No encontre consola_nba.html junto a claro_nexus.py", 500
            )
        self._enviar(
            archivo.read_bytes(),
            "text/html; charset=utf-8",
            extra={"Content-Disposition": "inline; filename=consola_nba.html"},
        )

    def _servir_config(self) -> None:
        with STATE_LOCK:
            campanas = STATE["campanas"]
            reglas = STATE["reglas"]
            parametros = STATE["parametros"]
            config_dir = STATE["config_dir"]
        if campanas is None:
            return self._error("Configuracion no cargada", 500)
        avisos = validar_configuracion(campanas, reglas, parametros)
        self._json(
            {
                "ok": True,
                "config_dir": str(config_dir),
                "dataset": STATE["dataset"],
                "fecha": STATE["fecha"],
                "campanas": df_registros(campanas),
                "reglas": df_registros(reglas),
                "parametros": df_registros(parametros),
                "avisos": avisos,
                "ortools": ORTOOLS_OK,
                "datasets": self._lista_datasets(),
                "historial": STATE["historial"],
                "columnas": {
                    "campanas": CAMPANAS_COLUMNS,
                    "reglas": REGLAS_COLUMNS,
                    "parametros": PARAMETROS_COLUMNS,
                },
            }
        )

    def _lista_datasets(self) -> List[Dict[str, Any]]:
        config_dir = Path(STATE["config_dir"] or APP_DIR)
        salida = []
        for clave, (nombre, etiqueta) in DATASETS.items():
            ruta = config_dir / nombre
            salida.append(
                {
                    "id": clave,
                    "archivo": nombre,
                    "etiqueta": etiqueta,
                    "existe": ruta.exists(),
                }
            )
        return salida

    def _servir_datasets(self) -> None:
        self._json({"ok": True, "datasets": self._lista_datasets()})

    def _servir_parrilla(self) -> None:
        with STATE_LOCK:
            csv = STATE["parrilla_csv"]
            resumen = STATE["resumen"]
        if not csv:
            return self._error("Todavia no hay una parrilla generada. Ejecute el motor.", 404)
        nombre = f"CLARO_NEXUS_parrilla_{resumen.get('dataset', DEFAULT_DATASET)}.csv"
        self._enviar(
            csv,
            "text/csv; charset=utf-8",
            extra={"Content-Disposition": f'attachment; filename="{nombre}"'},
        )

    def _manejar_run(self, body: Dict[str, Any]) -> None:
        try:
            with STATE_LOCK:
                resumen = ejecutar_desde_consola(body)
            self._json(resumen)
        except Exception as exc:
            self._error(str(exc), 400, traceback.format_exc(limit=4))

    def _manejar_reset(self) -> None:
        """Vuelve a leer los CSV originales del proyecto."""
        try:
            config_dir = Path(STATE["config_dir"] or localizar_config_dir())
            campanas, reglas, parametros = cargar_configuracion(config_dir)
            with STATE_LOCK:
                STATE["campanas"] = campanas
                STATE["reglas"] = reglas
                STATE["parametros"] = parametros
            self._servir_config()
        except Exception as exc:
            self._error(str(exc), 400, traceback.format_exc(limit=4))

    def _manejar_save(self, body: Dict[str, Any]) -> None:
        try:
            if body.get("campanas") is not None:
                campanas, reglas, parametros = _paquetes_config(body)
            else:
                with STATE_LOCK:
                    campanas = STATE["campanas"].copy(deep=True)
                    reglas = STATE["reglas"].copy(deep=True)
                    parametros = STATE["parametros"].copy(deep=True)
            avisos = validar_configuracion(campanas, reglas, parametros)
            config_dir = Path(STATE["config_dir"] or localizar_config_dir())
            nombre = str(body.get("nombre") or datetime.now().strftime("%Y%m%d_%H%M%S"))
            nombre = re.sub(r"[^A-Za-z0-9_\-]", "_", nombre).strip("_") or "config"
            destino = config_dir / "config_guardada" / nombre
            guardar_configuracion(destino, campanas, reglas, parametros)
            with STATE_LOCK:
                STATE["campanas"] = _normalizar_campanas(campanas)
                STATE["reglas"] = _normalizar_reglas(reglas)
                STATE["parametros"] = _normalizar_parametros(parametros)
            self._json(
                {
                    "ok": True,
                    "ruta": str(destino),
                    "avisos": avisos,
                    "mensaje": f"Configuracion guardada en {destino}",
                }
            )
        except Exception as exc:
            self._error(str(exc), 400, traceback.format_exc(limit=4))


class NexusHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def servir(config_dir: Optional[str], dataset: str, fecha: str, host: str, puerto: int, abrir: bool) -> None:
    ruta = _cargar_estado_inicial(config_dir, dataset, fecha)
    url = f"http://{host}:{puerto}/"
    print("=" * 72)
    print(" CLARO NEXUS - Consola de configuracion")
    print("=" * 72)
    print(f" Configuracion : {ruta}")
    print(f" Dataset       : {dataset}")
    print(f" Fecha         : {fecha}")
    print(f" OR-Tools      : {'disponible' if ORTOOLS_OK else 'no instalado (se usara heuristica)'}")
    print(f" Consola       : {url}")
    print(" Detener       : Ctrl+C")
    print("=" * 72)
    try:
        servidor = NexusHTTPServer((host, puerto), NexusHTTPHandler)
    except OSError as exc:
        print(
            f"ERROR: no se pudo abrir {host}:{puerto} ({exc}). "
            "Pruebe con --port 8766.",
            file=sys.stderr,
        )
        return
    if abrir:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        servidor.server_close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _imprimir_metricas(metricas: Dict[str, Any]) -> None:
    print("")
    print("METRICAS")
    print("-" * 60)
    print(f"  Clientes              : {metricas['clientes']:,}")
    print(f"  Clientes asignados    : {metricas['clientes_asignados']:,}")
    print(f"  Cobertura             : {metricas['cobertura_pct']}%")
    print(f"  Productos asignados   : {metricas['productos_asignados']:,}")
    print(f"  Valor esperado        : {metricas['valor_esperado']}")
    print(f"  Jerarquia elegibles   : {metricas.get('jerarquia_pct_elegibles')}")
    print(f"  Jerarquia asignados   : {metricas.get('jerarquia_pct_asignados')}")
    print(f"  Violaciones R01-R16   : {metricas['violaciones']}")
    print(f"  Tiempo                : {metricas['tiempo_s']} s")
    print(f"  Motor                 : {metricas.get('motor')}")


def comando_run(args: argparse.Namespace) -> int:
    config_dir = localizar_config_dir(args.config_dir)
    campanas, reglas, parametros = cargar_configuracion(config_dir)
    datos = cargar_datos(config_dir, args.dataset)
    fecha = args.fecha or DEFAULT_FECHA
    pd.Timestamp(fecha)

    print(f"Configuracion : {config_dir}")
    print(f"Dataset       : {args.dataset} ({len(datos):,} clientes)")
    print(f"Fecha         : {fecha}")
    print(f"OR-Tools      : {'disponible' if ORTOOLS_OK else 'no instalado (heuristica)'}")

    resultado = ejecutar_orquestador(datos, campanas, reglas, parametros, fecha)
    _imprimir_metricas(resultado["metricas"])

    avisos = validar_configuracion(campanas, reglas, parametros)
    if avisos:
        print("")
        print("AVISOS DE CONFIGURACION")
        print("-" * 60)
        for aviso in avisos:
            print(f"  - {aviso}")

    detalle = resultado["validacion"]
    if not detalle.empty:
        print("")
        print("VIOLACIONES")
        print("-" * 60)
        print(detalle["restriccion"].value_counts().to_string())

    salida = Path(args.out) if args.out else (
        Path(DEFAULT_OUT) if args.dataset == DEFAULT_DATASET
        else config_dir / f"CLARO_NEXUS_parrilla_{args.dataset}.csv"
    )
    salida = salida.expanduser().resolve()
    salida.parent.mkdir(parents=True, exist_ok=True)
    resultado["parrilla"].to_csv(salida, index=False, encoding="utf-8-sig")
    print("")
    print(f"Parrilla guardada en: {salida}")
    return 0


def comando_validate(args: argparse.Namespace) -> int:
    config_dir = localizar_config_dir(args.config_dir)
    campanas, reglas, parametros = cargar_configuracion(config_dir)
    avisos = validar_configuracion(campanas, reglas, parametros)
    print(f"Configuracion: {config_dir}")
    print(f"Campanas: {len(campanas)} | Reglas: {len(reglas)} | Parametros: {len(parametros)}")
    if not avisos:
        print("Sin avisos: la configuracion es coherente.")
        return 0
    print("")
    for aviso in avisos:
        print(f"  - {aviso}")
    return 0


def comando_serve(args: argparse.Namespace) -> int:
    servir(args.config_dir, args.dataset, args.fecha or DEFAULT_FECHA, args.host, args.port, args.open)
    return 0


def comando_config(args: argparse.Namespace) -> int:
    config_dir = localizar_config_dir(args.config_dir)
    campanas, reglas, parametros = cargar_configuracion(config_dir)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print(f"Configuracion: {config_dir}\n")
    print("CAMPANAS")
    print(campanas.to_string(index=False))
    print("\nREGLAS")
    print(reglas.to_string(index=False))
    print("\nPARAMETROS")
    print(parametros.to_string(index=False))
    return 0


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claro_nexus.py",
        description="CLARO NEXUS - Orquestador Next Best Action configurable",
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    def comunes(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config-dir", help="Carpeta con config_campanas.csv, config_reglas.csv y config_parametros.csv")
        p.add_argument("--fecha", default=DEFAULT_FECHA, help=f"Fecha de evaluacion (YYYY-MM-DD, por defecto {DEFAULT_FECHA})")

    p_run = sub.add_parser("run", help="Ejecuta el motor y exporta la parrilla")
    comunes(p_run)
    p_run.add_argument("--dataset", default=DEFAULT_DATASET, choices=["demo", "full", "synthetic"], help="Dataset a usar (por defecto full)")
    p_run.add_argument("--out", help="Ruta del CSV de salida")
    p_run.set_defaults(func=comando_run)

    p_serve = sub.add_parser("serve", help="Levanta la consola web local")
    comunes(p_serve)
    p_serve.add_argument("--dataset", default=DEFAULT_DATASET, choices=["demo", "full", "synthetic"])
    p_serve.add_argument("--host", default=DEFAULT_HOST)
    p_serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_serve.add_argument("--open", action="store_true", help="Abre la consola en el navegador")
    p_serve.set_defaults(func=comando_serve)

    p_val = sub.add_parser("validate", help="Revisa la coherencia de la configuracion")
    comunes(p_val)
    p_val.set_defaults(func=comando_validate)

    p_cfg = sub.add_parser("config", help="Muestra la configuracion cargada")
    comunes(p_cfg)
    p_cfg.set_defaults(func=comando_config)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = construir_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    # Sin subcomando (o solo con opciones) se ejecuta "run" con la base completa.
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv = ["run"] + argv
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nCancelado.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if getattr(args, "debug", False):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())



