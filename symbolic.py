"""Strato di reasoning simbolico: sostituisce il safety-override if e valida l'NLP.

Due responsabilità dell'ontologia (non un dizionario, non un `if` scritto a mano):

  is_critical_state(mean_hrr) -> bool
      La soglia di sicurezza (0.90) NON è una costante Python: è dichiarata come dato
      nell'ontologia (ar:CriticalState ar:hasThreshold ...) in
      ontology/runner_state.owl. Si inietta la HRR misurata come tripla RDF e una
      query SPARQL confronta i due valori dentro il grafo — è la QUERY a decidere,
      non un confronto scritto a mano in controller.py. Solo rdflib, nessuna
      dipendenza Java.

  validate_speed(speed_kmh) -> bool
      Valida un numero estratto dal testo (intent.parse_numbers) contro il Constraint
      Gate SHACL in ontology/nlp_shapes.ttl (pyshacl): scarta velocità
      fisiologicamente assurde prima che arrivino a bpm_from_speed.
"""
from __future__ import annotations

from pathlib import Path

from rdflib import RDF, XSD, Graph, Literal, Namespace

BASE = Path(__file__).parent
RUNNER_STATE_OWL = BASE / "ontology" / "runner_state.owl"
NLP_SHAPES_TTL = BASE / "ontology" / "nlp_shapes.ttl"

AR = Namespace("http://runmaxxin.org/ontology#")

_CRITICAL_QUERY = """
PREFIX ar: <http://runmaxxin.org/ontology#>
ASK {
    ar:CriticalState ar:hasThreshold ?threshold .
    ?observation ar:hasCurrentHRR ?hrr .
    FILTER(?hrr >= ?threshold)
}
"""


def is_critical_state(mean_hrr: float) -> bool:
    """True se la HRR misurata supera la soglia dichiarata NELL'ONTOLOGIA (query SPARQL)."""
    graph = Graph().parse(RUNNER_STATE_OWL, format="turtle")
    observation = AR[f"obs_{abs(hash(float(mean_hrr)))}"]
    graph.add((observation, RDF.type, AR.RunnerState))
    graph.add((observation, AR.hasCurrentHRR, Literal(float(mean_hrr), datatype=XSD.float)))
    return bool(graph.query(_CRITICAL_QUERY).askAnswer)


def validate_speed(speed_kmh: float) -> bool:
    """True se la velocità rispetta il Constraint Gate SHACL (0, 45] km/h."""
    from pyshacl import validate

    data = Graph()
    observation = AR[f"speed_{abs(hash(float(speed_kmh)))}"]
    data.add((observation, RDF.type, AR.ExtractedSpeed))
    data.add((observation, AR.speedKmh, Literal(float(speed_kmh), datatype=XSD.float)))

    shapes = Graph().parse(NLP_SHAPES_TTL, format="turtle")

    conforms, _report_graph, _report_text = validate(
        data_graph=data,
        shacl_graph=shapes,
    )
    return bool(conforms)


if __name__ == "__main__":
    print("== is_critical_state (SPARQL, soglia dichiarata nell'ontologia) ==")
    for hrr in (0.60, 0.86, 0.90, 0.95):
        print(f"  HRR={hrr} -> critical={is_critical_state(hrr)}")

    print("\n== validate_speed (SHACL, pyshacl) ==")
    for v in (12.0, 40.0, 45.0, 45.1, 300.0, -5.0):
        print(f"  speed={v} km/h -> valid={validate_speed(v)}")
