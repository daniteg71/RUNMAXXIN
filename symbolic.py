#  il seguente codice rappresenta lo strato di reasoning simbolico del progetto:
#  sostituisce il safety-override "if" scritto a mano e valida l'output dell'NLP,
#  delegando le decisioni a query sull'ontologia invece che a costanti Python.

#  le tre funzioni definite lavorano in questo modo:

#    is_critical_state(mean_hrr) -> bool
#        La soglia di sicurezza (0.90) NON è una costante Python: è dichiarata come dato
#        nell'ontologia (ar:CriticalState ar:hasThreshold ...) in
#        ontology/runner_state.owl. Si inietta la HRR misurata come tripla RDF e una
#        query SPARQL confronta i due valori dentro il grafo — è la QUERY a decidere,
#        non un confronto scritto a mano in controller.py. Solo rdflib, nessuna
#        dipendenza Java.

#    validate_speed(speed_kmh) -> bool
#        Valida un numero estratto dal testo (intent.parse_numbers) contro il Constraint
#        Gate SHACL in ontology/nlp_shapes.ttl (pyshacl): scarta velocità
#        fisiologicamente assurde prima che arrivino a bpm_from_speed.

#    is_effort_compatible(song_efforts, effort_band) -> bool
#        Pattern Generator->Validator: il recommender sovra-genera Top-K candidati per
#        vicinanza vettoriale, che possono ignorare la compatibilità di sforzo (misurato:
#        16.7% di violazioni). Questa funzione valida — via query SPARQL, stesso meccanismo
#        di is_critical_state — se le etichette matches_effort della canzone intersecano
#        la effort_band del target; chi chiama scorre il Top-K e tiene la prima compatibile.


#  vengono utilizzate le librerie:
#    - annotations per rendere più flessibili le annotazioni di tipo
#    - path per gestire i percorsi dei file ontologici
#    - rdflib per lavorare con grafi RDF e query SPARQL
#    - pyshacl (importato dentro validate_speed) per la validazione SHACL


from __future__ import annotations
from pathlib import Path
from rdflib import RDF, XSD, Graph, Literal, Namespace


#percorsi dei file ontologici
BASE = Path(__file__).parent
RUNNER_STATE_OWL = BASE / "ontology" / "runner_state.owl"
NLP_SHAPES_TTL = BASE / "ontology" / "nlp_shapes.ttl"

#namespace del progetto, usato in tutte le triple RDF
AR = Namespace("http://runmaxxin.org/ontology#")


#  query SPARQL ASK: risponde True se esiste una osservazione la cui HRR
#  supera (>=) la soglia dichiarata dalla classe CriticalState nell'ontologia
_CRITICAL_QUERY = """
PREFIX ar: <http://runmaxxin.org/ontology#>
ASK {
    ar:CriticalState ar:hasThreshold ?threshold .
    ?observation ar:hasCurrentHRR ?hrr .
    FILTER(?hrr >= ?threshold)
}
"""


#  la seguente funzione verifica se l'utente si trova in uno stato cardiaco critico.
#  carica l'ontologia, inietta la HRR misurata come tripla RDF e lascia decidere alla
#  query SPARQL: è il grafo, non un if scritto a mano, a stabilire la soglia.
def is_critical_state(mean_hrr: float) -> bool:
    graph = Graph().parse(RUNNER_STATE_OWL, format="turtle")
    observation = AR[f"obs_{abs(hash(float(mean_hrr)))}"]
    graph.add((observation, RDF.type, AR.RunnerState))
    graph.add((observation, AR.hasCurrentHRR, Literal(float(mean_hrr), datatype=XSD.float)))
    return bool(graph.query(_CRITICAL_QUERY).askAnswer)


#  query SPARQL ASK: risponde True se canzone e target condividono almeno
#  un'etichetta di sforzo (intersezione non vuota fra hasEffort e allowsEffort)
_EFFORT_COMPATIBLE_QUERY = """
PREFIX ar: <http://runmaxxin.org/ontology#>
ASK {
    ?song ar:hasEffort ?e .
    ?target ar:allowsEffort ?e .
}
"""


#  la seguente funzione valida la compatibilità di sforzo fra una canzone e il target.
#  costruisce un piccolo grafo con le etichette della canzone (hasEffort) e quelle
#  ammesse dal target (allowsEffort) e chiede alla query se hanno un'etichetta in comune.
#  è il passo "Validator" del pattern Generator->Validator applicato al Top-K del recommender.
def is_effort_compatible(song_efforts: list[str], effort_band: tuple[str, ...]) -> bool:
    graph = Graph()
    song, target = AR.song, AR.target
    for e in song_efforts:
        graph.add((song, AR.hasEffort, AR[e]))
    for e in effort_band:
        graph.add((target, AR.allowsEffort, AR[e]))
    return bool(graph.query(_EFFORT_COMPATIBLE_QUERY).askAnswer)


#  la seguente funzione valida il parametro velocità estratto dall'NLP.
#  True se la velocità rispetta il Constraint Gate SHACL (0, 45] km/h dichiarato in nlp_shapes.ttl.
#  pyshacl viene importato qui dentro perché è una dipendenza pesante usata solo da questa funzione.
def validate_speed(speed_kmh: float) -> bool:
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


# simulazione
if __name__ == "__main__":
    print("== is_critical_state (SPARQL, soglia dichiarata nell'ontologia) ==")
    for hrr in (0.60, 0.86, 0.90, 0.95):
        print(f"  HRR={hrr} -> critical={is_critical_state(hrr)}")

    print("\n== validate_speed (SHACL, pyshacl) ==")
    for v in (12.0, 40.0, 45.0, 45.1, 300.0, -5.0):
        print(f"  speed={v} km/h -> valid={validate_speed(v)}")

    print("\n== is_effort_compatible (SPARQL, gate sul Top-K) ==")
    print("  HighEffort   vs (TargetEffort,)        ->", is_effort_compatible(["HighEffort"], ("TargetEffort",)))
    print("  TargetEffort vs (TargetEffort,)        ->", is_effort_compatible(["TargetEffort"], ("TargetEffort",)))
    print("  [Low,HighEffort] vs (High,VeryHigh)    ->", is_effort_compatible(["Low", "HighEffort"], ("HighEffort", "VeryHighEffort")))
