from beamter.ingest import Page, chunk_pages, clean_text, extract_units, is_heading, split_sentences


def test_sentence_split_protects_abbreviations_and_dates():
    text = "Bringen Sie z. B. den Pass mit. Am 1. Januar beginnt die Frist. Siehe Abs. 2 und Nr. 5."
    assert split_sentences(text) == [
        "Bringen Sie z. B. den Pass mit.",
        "Am 1. Januar beginnt die Frist.",
        "Siehe Abs. 2 und Nr. 5.",
    ]


def test_heading_detection():
    assert is_heading("ANMELDUNG")
    assert is_heading("1.2 Fristen und Gebühren")
    assert not is_heading("14. Tage")
    assert not is_heading("Sie müssen sich anmelden.")
    assert not is_heading("")


def test_clean_text_joins_hyphenated_breaks():
    assert clean_text("Melde-\ngesetz   gilt") == "Meldegesetz gilt"


def test_sections_are_tracked_and_carried():
    units, last = extract_units("ANMELDUNG\nErster Satz. Zweiter Satz.\nFRISTEN\nDritter Satz.")
    assert [(s, t) for s, t in units] == [
        ("ANMELDUNG", "Erster Satz."),
        ("ANMELDUNG", "Zweiter Satz."),
        ("FRISTEN", "Dritter Satz."),
    ]
    assert last == "FRISTEN"
    carried, _ = extract_units("Noch ein Satz.", carry_section="FRISTEN")
    assert carried[0][0] == "FRISTEN"


def test_chunks_respect_size_never_cross_pages_and_keep_whole_sentences():
    sentence = "Dies ist ein vollständiger Beispielsatz mit genug Wörtern."
    page_text = " ".join([sentence] * 30)
    pages = [Page("a.pdf", 1, page_text), Page("a.pdf", 2, page_text)]
    chunks = chunk_pages(pages, max_chars=300, overlap_sentences=1, min_chars=100)
    assert chunks
    assert {c.page for c in chunks} == {1, 2}
    for c in chunks:
        assert len(c.text) <= 300
        assert c.text.endswith(".")  # ends on a sentence boundary
    first_page = [c for c in chunks if c.page == 1]
    assert first_page[0].text.split(".")[-2:]  # sanity
    # overlap: the last sentence of a chunk reappears at the start of the next one
    assert first_page[1].text.startswith(sentence)


def test_oversized_sentence_is_hard_split():
    pages = [Page("a.txt", 1, "wort " * 400 + "ende.")]
    chunks = chunk_pages(pages, max_chars=200, overlap_sentences=0, min_chars=50)
    assert len(chunks) > 5
    assert all(len(c.text) <= 200 for c in chunks)


def test_ids_are_stable_and_unique():
    pages = [Page("a.txt", 1, "Satz eins. " * 80)]
    a = chunk_pages(pages, max_chars=200, overlap_sentences=1, min_chars=50)
    b = chunk_pages(pages, max_chars=200, overlap_sentences=1, min_chars=50)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


def test_embed_text_carries_title_and_section():
    chunk = chunk_pages([Page("anmeldung_merkblatt.pdf", 1, "ANMELDUNG\nSie haben 14 Tage Zeit.")])[0]
    assert chunk.embed_text.startswith("anmeldung merkblatt - ANMELDUNG\n")


# ── additions for the cloud version ──────────────────────────────────────

def test_payload_roundtrip_preserves_chunk():
    from beamter.ingest import Chunk

    chunk = chunk_pages([Page("a.pdf", 3, "ANMELDUNG\nSie haben 14 Tage Zeit.")])[0]
    assert Chunk.from_payload(chunk.id, chunk.to_payload()) == chunk
    assert "id" not in chunk.to_payload()  # the id is the Qdrant point id


def test_fingerprint_changes_with_content_and_settings():
    from beamter.ingest import fingerprint

    pages = [Page("a.txt", 1, "Text eins.")]
    base = fingerprint(pages, {"chunk_chars": 700})
    assert base == fingerprint(pages, {"chunk_chars": 700})
    assert base != fingerprint(pages, {"chunk_chars": 500})
    assert base != fingerprint([Page("a.txt", 1, "Text zwei.")], {"chunk_chars": 700})


def test_load_pages_splits_text_files_on_form_feed(tmp_path):
    from beamter.ingest import load_pages

    (tmp_path / "doc.txt").write_text("Seite eins.\fSeite zwei.\f   ", encoding="utf-8")
    (tmp_path / "ignored.docx").write_text("x", encoding="utf-8")
    pages = load_pages(tmp_path)
    assert [(p.source, p.page, p.text) for p in pages] == [
        ("doc.txt", 1, "Seite eins."),
        ("doc.txt", 2, "Seite zwei."),
    ]


def test_readme_and_sources_notes_are_not_indexed(tmp_path):
    from beamter.ingest import load_pages

    (tmp_path / "README.md").write_text("Hinweise zum Ordner.", encoding="utf-8")
    (tmp_path / "SOURCES.md").write_text("Quellenliste.", encoding="utf-8")
    (tmp_path / "anmeldung.txt").write_text("Echter Inhalt.", encoding="utf-8")
    assert [p.source for p in load_pages(tmp_path)] == ["anmeldung.txt"]


def test_unreadable_pdf_is_skipped_not_fatal(tmp_path, caplog):
    from beamter.ingest import load_pages

    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 this is not a real pdf")
    (tmp_path / "fine.txt").write_text("Gueltiger Inhalt.", encoding="utf-8")
    with caplog.at_level("WARNING"):
        pages = load_pages(tmp_path)
    assert [p.source for p in pages] == ["fine.txt"]
    assert "broken.pdf could not be read" in caplog.text


def test_callout_labels_are_not_headings():
    for label in ("HINWEIS", "BEISPIEL", "TIPP", "BITTE BEACHTEN SIE", "EUR"):
        assert not is_heading(label)
    units, last = extract_units("ANMELDUNG\nHINWEIS\nBringen Sie den Pass mit.")
    assert last == "ANMELDUNG"  # the callout did not replace the real section
    assert units[0][0] == "ANMELDUNG"


def test_amounts_and_stray_caps_are_not_headings():
    for line in ("2.500 Euro) oder", "15.500 Euro) besitzt", "ODER", "BGB", "UND"):
        assert not is_heading(line), line
    assert is_heading("3.5. Indexmiete")
    assert is_heading("2.3 Schutzfristen vor und nach der Geburt")


def test_section_labels_do_not_leak_across_pages():
    pages = [Page("a.pdf", 1, "1.1 Fristen\nDie Frist beträgt vierzehn Tage nach dem Einzug in die Wohnung."),
             Page("a.pdf", 2, "Dies ist eine andere Seite ohne eigene Überschrift, aber mit genug Text.")]
    chunks = chunk_pages(pages)
    assert chunks[0].section == "1.1 Fristen"
    assert chunks[1].section == ""  # not the stale heading from page 1
    assert chunk_pages(pages, carry_sections=True)[1].section == "1.1 Fristen"


def test_running_headers_are_dropped():
    pages = [Page("a.pdf", i, f"IN GERMANY\nSeite {i} hat einen eigenen Satz über Wohnungen und Mieten.")
             for i in range(1, 5)]
    chunks = chunk_pages(pages)
    assert all(c.section == "" and "IN GERMANY" not in c.text for c in chunks)


def test_tiny_fragments_are_merged_or_dropped():
    long_sentence = "Dies ist ein vollständiger Beispielsatz mit genug Wörtern und Inhalt. " * 6
    page = Page("a.pdf", 1, f"{long_sentence}\n\nSeite 3")
    texts = [c.text for c in chunk_pages([page], max_chars=700)]
    assert all(len(t) >= 60 for t in texts)
