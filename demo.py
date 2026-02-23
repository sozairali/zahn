"""
Demo: insert 10 sample conversations and process them with the worker.
"""
import textwrap
import psycopg

from zahn.config import load_settings
from zahn.db import get_connection, reset_stale_claims
from zahn.worker import run_one_iteration

SAMPLE_JOBS = [
    # ── English (5) ──────────────────────────────────────────────────────────
    {
        "message_text": (
            "Just wanted to say the crown on case #4821 came in perfect. "
            "The fit was spot on and my patient loved the shade match. "
            "Really appreciate the attention to detail — keep it up."
        ),
    },
    {
        "message_text": (
            "This is the third time I'm sending back the same bridge. "
            "The margins are still open and the occlusion is nowhere near right. "
            "I'm losing patience and my patient has been waiting six weeks."
        ),
    },
    {
        "message_text": (
            "Case #5503 was due yesterday and still hasn't arrived. "
            "I've left two voicemails and sent an email — nobody has gotten back to me. "
            "Patient is coming in tomorrow morning, this is completely unacceptable."
        ),
    },
    {
        "message_text": (
            "We received case #6102 this morning, everything looks great. "
            "Thank you for the quick turnaround, we'll have more cases coming your way."
        ),
    },
    {
        "message_text": (
            "I've been trying to reach someone at the lab for three days and no one "
            "is calling me back. The quality on the last two cases has been inconsistent "
            "and at this point I'm seriously considering switching labs."
        ),
    },
    # ── Spanish (3) ──────────────────────────────────────────────────────────
    {
        "message_text": (
            "El caso #7234 llegó con el color completamente equivocado. "
            "Ya es la segunda vez que pasa esto con el mismo paciente. "
            "Necesito que alguien me llame hoy mismo para resolver esto."
        ),
    },
    {
        "message_text": (
            "Quería agradecer el trabajo en el caso #8891. "
            "El ajuste fue perfecto y la paciente quedó muy contenta con el resultado. "
            "Seguiremos enviando casos con ustedes sin duda."
        ),
    },
    {
        "message_text": (
            "Necesito saber el estado del caso #9102. "
            "El paciente tiene cita mañana a las 10 y aún no hemos recibido el trabajo."
        ),
    },
    # ── French (2) ───────────────────────────────────────────────────────────
    {
        "message_text": (
            "Le bridge du dossier #3312 est arrivé avec une teinte incorrecte. "
            "C'est la deuxième fois en un mois et le délai de livraison était déjà trop long. "
            "Si la situation ne s'améliore pas rapidement, nous devrons chercher un autre laboratoire."
        ),
    },
    {
        "message_text": (
            "Bonjour, je voulais confirmer la bonne réception du cas #4450. "
            "La prothèse est impeccable et la patiente est vraiment ravie du résultat. "
            "Merci pour votre excellent travail, à très bientôt."
        ),
    },
]

LABEL_COLOUR = {
    "frustration": "\033[91m",  # red
    "satisfaction": "\033[92m",  # green
    "neutral":      "\033[93m",  # yellow
}
RESET = "\033[0m"


def insert_samples(conn: psycopg.Connection) -> list[int]:
    ids = []
    for job in SAMPLE_JOBS:
        cur = conn.execute(
            """
            INSERT INTO sentiment_jobs (message_text)
            VALUES (%(message_text)s)
            RETURNING id
            """,
            job,
        )
        ids.append(cur.fetchone()["id"])
    conn.commit()
    return ids


def print_results(conn: psycopg.Connection, job_ids: list[int]) -> None:
    rows = conn.execute(
        """
        SELECT id, detected_language, sentiment_label, excerpt, reasoning, status
        FROM sentiment_jobs
        WHERE id = ANY(%(ids)s)
        ORDER BY id
        """,
        {"ids": job_ids},
    ).fetchall()

    print("\n" + "=" * 72)
    print(f"{'ZAHN DEMO — RESULTS':^72}")
    print("=" * 72)

    for row in rows:
        label = row["status"] if row["sentiment_label"] is None else row["sentiment_label"]
        colour = LABEL_COLOUR.get(label, "")
        lang = (row["detected_language"] or "??").upper()

        print(f"\n[{lang}]  Job #{row['id']}  →  {colour}{label.upper()}{RESET}")

        if row["excerpt"]:
            print(f"  Excerpt  : \"{row['excerpt']}\"")
        if row["reasoning"]:
            wrapped = textwrap.fill(row["reasoning"], width=66, initial_indent="  Reasoning: ", subsequent_indent="             ")
            print(wrapped)

    print("\n" + "=" * 72)
    counts = {}
    for row in rows:
        lbl = row["sentiment_label"] or row["status"]
        counts[lbl] = counts.get(lbl, 0) + 1
    for lbl, n in sorted(counts.items()):
        colour = LABEL_COLOUR.get(lbl, "")
        print(f"  {colour}{lbl:<14}{RESET}  {n}")
    print("=" * 72 + "\n")


def main() -> None:
    config = load_settings()

    print("Inserting 10 sample conversations...")
    with get_connection(config) as conn:
        reset_stale_claims(conn)
        job_ids = insert_samples(conn)
    print(f"Inserted job IDs: {job_ids}")

    print("\nProcessing jobs...\n")
    processed = 0
    while True:
        did_work = run_one_iteration(config)
        if not did_work:
            break
        processed += 1
        print(f"  [{processed}/10] done")

    print(f"\nAll {processed} jobs processed.")

    with get_connection(config) as conn:
        print_results(conn, job_ids)


if __name__ == "__main__":
    main()
