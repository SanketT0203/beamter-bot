import json
KG=[("ba_merkblatt_kindergeld_2026.pdf","monatlich 259 Euro")]
MUT=[("bmfsfj_mutterschutz_schwangere.pdf","sechs Wochen vor der Entbindung"),("bmfsfj_mutterschutz_arbeitgeber.pdf","sechs Wochen vor der Entbindung")]
ST=[("bamf_bildung_und_beruf.pdf","120 Tage oder 240 halbe Tage"),("bamf_anwendungshinweise_bmi.pdf","120 Tage oder 240 halbe Tage")]
H20=[("daad_studying_in_germany.pdf","The limit is currently 20 hours per week")]
TUI=[("daad_studying_in_germany.pdf","You do not usually pay tuition fees at public higher education institutions")]
BC=[("blue_card_merkblatt.pdf","50.700,-EUR/Jahr")]
ARB=[("ba_merkblatt_fuer_arbeitslose.pdf","spätestens drei Monate vor der Beendigung")]
EG=[("bmfsfj_elterngeld_und_elternzeit.pdf","65 Prozent des Netto-Einkommens")]
WG=[("bmwsb_auf_wohnungssuche.pdf","DIESE UNTERLAGEN BENÖTIGEN SIE FÜR DEN WOHNGELDANTRAG")]
GF=[("bmf_steuern_von_a_bis_z.pdf","Grundfreibetrag in Höhe von 12.096 Euro")]
MJ=[("drv_minijobs_midijobs.pdf","603 Euro im Monat")]
def q(id,t,lang,question,ev=None):
    d={"id":id,"type":t,"lang":lang,"question":question,"answerable":ev is not None}
    d["evidence"]=[{"source":s,"span":p} for s,p in (ev or [])]
    return d
Q=[
q("de_kindergeld","factoid_de","de","Wie hoch ist das Kindergeld pro Kind im Monat 2026?",KG),
q("de_kaution","factoid_de","de","Wie hoch darf die Mietkaution höchstens sein?",[("bmjv_leitfaden_wohnraummietrecht.pdf","drei Monatsmieten")]),
q("de_mutterschutz","factoid_de","de","Wie lange dauern die Mutterschutzfristen vor und nach der Geburt?",MUT),
q("de_studierende_arbeit","factoid_de","de","Wie viele Tage im Jahr dürfen ausländische Studierende arbeiten?",ST),
q("de_rundfunk","factoid_de","de","Wie hoch ist der Rundfunkbeitrag pro Monat?",[("rundfunkbeitrag_befreiung_ermaessigung.pdf","18,36")]),
q("de_grundfreibetrag","factoid_de","de","Wie hoch ist der Grundfreibetrag?",GF),
q("de_integrationskurs","factoid_de","de","Was kostet der Integrationskurs pro Unterrichtsstunde?",[("bamf_integrationskurs_zulassung.pdf","2,29 Euro pro Unterrichtsstunde")]),
q("de_pflegegeld","factoid_de","de","Wie viel Pflegegeld bekommt man bei Pflegegrad 3?",[("bmg_ratgeber_pflegeleistungen.pdf","Pflegegrad 3 545 Euro")]),
q("de_minijob","factoid_de","de","Bis zu welchem Verdienst im Monat ist ein Minijob möglich?",MJ),
q("de_arbeitsuchend","factoid_de","de","Wann muss ich mich arbeitsuchend melden, wenn mein Arbeitsverhältnis endet?",ARB),
q("de_alg_anwartschaft","factoid_de","de","Wie lange muss ich versicherungspflichtig gewesen sein, um Arbeitslosengeld zu bekommen?",[("ba_arbeitslosengeld_ausland.pdf","mindestens 12 Monate versicherungspflichtig")]),
q("de_elterngeld","factoid_de","de","Wie viel Prozent des Nettoeinkommens beträgt das Basiselterngeld?",EG),
q("de_anmeldung","factoid_de","de","Innerhalb welcher Frist muss ich meinen Wohnsitz anmelden?",[("anmeldung_merkblatt.pdf","14 Tage Zeit")]),
q("de_bluecard","factoid_de","de","Welches Mindestgehalt wird für die Blaue Karte verlangt?",BC),
q("de_wohngeld","factoid_de","de","Welche Unterlagen brauche ich für den Wohngeldantrag?",WG),
q("de_fuehrerschein","factoid_de","de","Kann die Frist für den ausländischen Führerschein verlängert werden?",[("bremen_auslaendische_fahrerlaubnis.pdf","bis zu sechs Monate verlängern")]),
q("de_student_kv","factoid_de","de","Bis zu welchem Alter bieten gesetzliche Krankenkassen Studierenden einen günstigen Tarif an?",[("bamf_studie_internationale_studierende.pdf","bis zum Ende des 14. Fachsemesters")]),
q("en_kindergeld","english","en","How much is child benefit per month in 2026?",KG),
q("en_student_days","english","en","How many days per year may international students work?",ST),
q("en_20h","english","en","How many hours per week can a student work without German health insurance?",H20),
q("en_tuition","english","en","Do I have to pay tuition fees at public universities in Germany?",TUI),
q("en_semester_fee","english","en","How high is the semester fee at German universities?",[("daad_studying_in_germany.pdf","between €170 and €450")]),
q("en_bluecard","english","en","What is the minimum gross salary for the EU Blue Card?",BC),
q("en_jobseeker","english","en","How early do I have to register as job-seeking before my employment ends?",ARB),
q("en_parental","english","en","What share of my net income does parental allowance replace?",EG),
q("en_housing_benefit","english","en","Which documents do I need to apply for housing benefit?",WG),
q("x_de_en_20h","cross_lingual","de","Wie viele Wochenstunden dürfen Studierende ohne deutsche Krankenversicherung arbeiten?",H20),
q("x_de_en_tuition","cross_lingual","de","Muss ich an staatlichen Hochschulen Studiengebühren zahlen?",TUI),
q("x_en_de_mietkaution","cross_lingual","en","What is the maximum rental deposit a landlord may ask for?",[("bmjv_leitfaden_wohnraummietrecht.pdf","drei Monatsmieten")]),
q("x_en_de_maternity","cross_lingual","en","How many weeks before birth does maternity protection begin?",MUT),
q("kw_grundfreibetrag","keyword","de","Grundfreibetrag 2025",GF),
q("kw_geringfuegigkeit","keyword","de","Geringfügigkeitsgrenze",MJ),
q("kw_steuer_id","keyword","de","Steuer-Identifikationsnummer Antrag Arbeitslosengeld",[("ba_merkblatt_fuer_arbeitslose.pdf","Bitte geben Sie diese im Antrag auf Arbeitslosengeld an")]),
q("oos_bau","out_of_scope","de","Was kostet eine Baugenehmigung für ein Einfamilienhaus?"),
q("oos_hund","out_of_scope","de","Wie hoch ist die Hundesteuer in München?"),
q("oos_weather","out_of_scope","en","What is the weather like in Berlin tomorrow?"),
q("oos_recipe","out_of_scope","en","How do I cook Maultaschen?"),
]
json.dump({"version":1,"questions":Q},open("golden.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print(len(Q))
