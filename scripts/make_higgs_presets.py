"""Dev tool: build the bundled Higgs preset voices in vibetts/voices/higgs/<VOICE>/<lang>.*

For each preset (F1, F2, M1, M2) and each Supertonic language that Higgs also reads, a neutral
sentence of about 10 s is synthesised with the app's SupertonicEngine (same voice name), resampled
to 24 kHz and encoded with Higgs's audio_encoder: `<lang>.codes.npy` (int16 [8, T]) + `<lang>.txt`.

Steps (run from the repo root with the app venv, except `asr`):
  make   build the clips             python scripts/make_higgs_presets.py make
  check  Higgs test read per clip    python scripts/make_higgs_presets.py check [--seed-offset 1]
  asr    word-match ratio per read   <venv with faster-whisper> scripts/make_higgs_presets.py asr
  prune  drop clips below 0.6        python scripts/make_higgs_presets.py prune
  table  write the voice x language table to presets.md

State lives in <work>/results.json (default .claude/temp/vtf09a/check). Nothing is committed here.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

THRESHOLD = 0.6
VOICES = ("F1", "F2", "M1", "M2")

# Neutral clip sentences (about 10 s): "The morning train arrived a little earlier than expected,
# so we had time to buy a cup of coffee and talk about our plans for the weekend."
CLIP_TEXT = {
    "en": "The morning train arrived a little earlier than expected, so we had time to buy a cup of coffee and talk about our plans for the weekend.",
    "hr": "Jutarnji vlak stigao je malo ranije nego što smo očekivali, pa smo imali vremena kupiti kavu i razgovarati o planovima za vikend.",
    "sl": "Jutranji vlak je prispel malo prej, kot smo pričakovali, zato smo imeli čas kupiti kavo in se pogovoriti o načrtih za vikend.",
    "sk": "Ranný vlak prišiel o niečo skôr, ako sme čakali, takže sme mali čas kúpiť si kávu a porozprávať sa o plánoch na víkend.",
    "cs": "Ranní vlak přijel o něco dříve, než jsme čekali, takže jsme měli čas koupit si kávu a popovídat si o plánech na víkend.",
    "pl": "Poranny pociąg przyjechał nieco wcześniej, niż się spodziewaliśmy, więc mieliśmy czas kupić kawę i porozmawiać o planach na weekend.",
    "de": "Der Frühzug kam etwas früher als erwartet, also hatten wir Zeit, einen Kaffee zu kaufen und über unsere Pläne für das Wochenende zu sprechen.",
    "es": "El tren de la mañana llegó un poco antes de lo previsto, así que tuvimos tiempo de comprar un café y hablar de nuestros planes para el fin de semana.",
    "it": "Il treno del mattino è arrivato un po' prima del previsto, così abbiamo avuto il tempo di comprare un caffè e parlare dei nostri programmi per il fine settimana.",
    "fr": "Le train du matin est arrivé un peu plus tôt que prévu, alors nous avons eu le temps d'acheter un café et de parler de nos projets pour le week-end.",
    "pt": "O comboio da manhã chegou um pouco mais cedo do que o esperado, por isso tivemos tempo de comprar um café e falar sobre os nossos planos para o fim de semana.",
    "nl": "De ochtendtrein kwam iets eerder aan dan verwacht, dus we hadden tijd om een kop koffie te kopen en over onze plannen voor het weekend te praten.",
    "da": "Morgentoget ankom lidt tidligere end forventet, så vi havde tid til at købe en kop kaffe og tale om vores planer for weekenden.",
    "sv": "Morgontåget kom lite tidigare än väntat, så vi hade tid att köpa en kopp kaffe och prata om våra planer för helgen.",
    "fi": "Aamujuna saapui hieman odotettua aikaisemmin, joten meillä oli aikaa ostaa kahvia ja puhua viikonlopun suunnitelmista.",
    "et": "Hommikune rong saabus veidi oodatust varem, nii et meil oli aega osta tass kohvi ja rääkida nädalavahetuse plaanidest.",
    "lv": "Rīta vilciens ieradās nedaudz agrāk, nekā gaidījām, tāpēc mums bija laika nopirkt kafiju un parunāt par plāniem nedēļas nogalei.",
    "lt": "Rytinis traukinys atvyko kiek anksčiau, nei tikėjomės, todėl turėjome laiko nusipirkti kavos ir pasikalbėti apie savaitgalio planus.",
    "hu": "A reggeli vonat kicsit korábban érkezett a vártnál, így volt időnk venni egy kávét, és beszélgetni a hétvégi terveinkről.",
    "ro": "Trenul de dimineață a sosit puțin mai devreme decât ne așteptam, așa că am avut timp să cumpărăm o cafea și să vorbim despre planurile noastre pentru weekend.",
    "bg": "Сутрешният влак пристигна малко по-рано от очакваното, така че имахме време да купим кафе и да поговорим за плановете си за уикенда.",
    "ru": "Утренний поезд прибыл немного раньше, чем мы ожидали, поэтому у нас было время купить кофе и поговорить о планах на выходные.",
    "uk": "Ранковий потяг прибув трохи раніше, ніж ми очікували, тож у нас був час купити каву й поговорити про плани на вихідні.",
    "el": "Το πρωινό τρένο έφτασε λίγο νωρίτερα από το αναμενόμενο, οπότε είχαμε χρόνο να αγοράσουμε έναν καφέ και να μιλήσουμε για τα σχέδιά μας για το Σαββατοκύριακο.",
    "tr": "Sabah treni beklediğimizden biraz erken geldi, bu yüzden bir kahve alıp hafta sonu planlarımız hakkında konuşacak vaktimiz oldu.",
    "vi": "Chuyến tàu buổi sáng đến sớm hơn dự kiến một chút, nên chúng tôi có thời gian mua một ly cà phê và nói về kế hoạch cuối tuần.",
    "id": "Kereta pagi tiba sedikit lebih awal dari perkiraan, jadi kami punya waktu untuk membeli secangkir kopi dan membicarakan rencana akhir pekan.",
    "hi": "सुबह की ट्रेन उम्मीद से थोड़ी जल्दी पहुँच गई, इसलिए हमारे पास एक कप कॉफ़ी खरीदने और सप्ताहांत की योजनाओं के बारे में बात करने का समय था।",
    "ar": "وصل قطار الصباح قبل الموعد المتوقع بقليل، فكان لدينا وقت لشراء كوب من القهوة والحديث عن خططنا لعطلة نهاية الأسبوع.",
    "ko": "아침 기차가 예상보다 조금 일찍 도착해서, 우리는 커피를 한 잔 사고 주말 계획에 대해 이야기할 시간이 있었다.",
}

# Test-read sentences (a different, shorter text): "The library opens at nine, and the quiet
# reading room is on the second floor."
TEST_TEXT = {
    "en": "The library opens at nine, and the quiet reading room is on the second floor.",
    "hr": "Knjižnica se otvara u devet sati, a tiha čitaonica nalazi se na drugom katu.",
    "sl": "Knjižnica se odpre ob devetih, tiha čitalnica pa je v drugem nadstropju.",
    "sk": "Knižnica sa otvára o deviatej a tichá čitáreň je na druhom poschodí.",
    "cs": "Knihovna se otevírá v devět hodin a tichá čítárna je ve druhém patře.",
    "pl": "Biblioteka otwiera się o dziewiątej, a cicha czytelnia znajduje się na drugim piętrze.",
    "de": "Die Bibliothek öffnet um neun Uhr, und der ruhige Lesesaal befindet sich im zweiten Stock.",
    "es": "La biblioteca abre a las nueve y la tranquila sala de lectura está en el segundo piso.",
    "it": "La biblioteca apre alle nove e la tranquilla sala di lettura si trova al secondo piano.",
    "fr": "La bibliothèque ouvre à neuf heures, et la salle de lecture calme se trouve au deuxième étage.",
    "pt": "A biblioteca abre às nove horas e a sala de leitura silenciosa fica no segundo andar.",
    "nl": "De bibliotheek gaat om negen uur open en de rustige leeszaal is op de tweede verdieping.",
    "da": "Biblioteket åbner klokken ni, og den stille læsesal ligger på anden sal.",
    "sv": "Biblioteket öppnar klockan nio, och det tysta läsrummet ligger på andra våningen.",
    "fi": "Kirjasto avautuu yhdeksältä, ja hiljainen lukusali on toisessa kerroksessa.",
    "et": "Raamatukogu avatakse üheksa ajal ja vaikne lugemissaal asub teisel korrusel.",
    "lv": "Bibliotēka atveras deviņos, un klusā lasītava atrodas otrajā stāvā.",
    "lt": "Biblioteka atidaroma devintą valandą, o tyli skaitykla yra antrame aukšte.",
    "hu": "A könyvtár kilenckor nyit, a csendes olvasóterem pedig a második emeleten van.",
    "ro": "Biblioteca se deschide la ora nouă, iar sala de lectură liniștită este la etajul al doilea.",
    "bg": "Библиотеката отваря в девет часа, а тихата читалня е на втория етаж.",
    "ru": "Библиотека открывается в девять часов, а тихий читальный зал находится на втором этаже.",
    "uk": "Бібліотека відчиняється о дев'ятій, а тиха читальна зала розташована на другому поверсі.",
    "el": "Η βιβλιοθήκη ανοίγει στις εννέα και η ήσυχη αίθουσα ανάγνωσης βρίσκεται στον δεύτερο όροφο.",
    "tr": "Kütüphane saat dokuzda açılır ve sessiz okuma salonu ikinci kattadır.",
    "vi": "Thư viện mở cửa lúc chín giờ, và phòng đọc yên tĩnh nằm ở tầng hai.",
    "id": "Perpustakaan buka pukul sembilan, dan ruang baca yang tenang ada di lantai dua.",
    "hi": "पुस्तकालय नौ बजे खुलता है, और शांत पठन कक्ष दूसरी मंज़िल पर है।",
    "ar": "تفتح المكتبة في التاسعة، وتقع قاعة القراءة الهادئة في الطابق الثاني.",
    "ko": "도서관은 아홉 시에 문을 열고, 조용한 독서실은 이 층에 있다.",
}
# Order to work through when time is short.
PRIORITY = ("hr", "en", "de", "sl", "sk", "cs", "pl", "es", "it", "fr")


def languages(selected: list[str] | None) -> list[str]:
    from vibetts.config import HIGGS_LANGUAGES, SUPERTONIC_LANGUAGES

    avail = [c for c in SUPERTONIC_LANGUAGES if c in HIGGS_LANGUAGES and c in CLIP_TEXT]
    order = list(avail)
    avail.sort(key=lambda c: (PRIORITY.index(c) if c in PRIORITY else len(PRIORITY), order.index(c)))
    return [c for c in avail if not selected or c in selected]


def write_wav(path: Path, samples, rate: int) -> None:
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def load_results(work: Path) -> dict:
    f = work / "results.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def save_results(work: Path, res: dict) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                       encoding="utf-8")


def higgs_engine(args):
    from vibetts import higgs

    return higgs.HiggsEngine(Path(args.higgs_dir), Path(args.voices_dir), threads=args.threads)


def cmd_make(args) -> None:
    from vibetts import higgs
    from vibetts.supertonic import SupertonicEngine
    from vibetts.tts import resample

    sup = SupertonicEngine(Path(args.supertonic_dir))
    eng = higgs_engine(args)
    out = higgs.presets_dir()
    work = Path(args.work)
    (work / "clips").mkdir(parents=True, exist_ok=True)
    for voice in args.voices:
        for lang in languages(args.langs):
            folder = out / voice
            if (folder / f"{lang}.codes.npy").exists() and not args.force:
                continue
            text = CLIP_TEXT[lang]
            audio = sup.synth(text, f"supertonic:{voice}", 1.0, lang)
            a24 = resample(audio, 44100, 24000)
            wav = work / "clips" / f"{voice}_{lang}.wav"
            write_wav(wav, a24, 24000)
            import numpy as np

            codes = eng.encode_clip(wav).astype(np.int16)
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / f"{lang}.codes.npy", codes)
            (folder / f"{lang}.txt").write_text(text, encoding="utf-8")
            print(f"{voice}/{lang}: {len(a24) / 24000:.1f} s, codes {codes.shape}", flush=True)


def cmd_check(args) -> None:
    import time

    from vibetts import higgs

    eng = higgs_engine(args)
    work = Path(args.work)
    (work / "reads").mkdir(parents=True, exist_ok=True)
    res = load_results(work)
    for lang in languages(args.langs):
        for voice in args.voices:
            key = f"{voice}/{lang}"
            if not (higgs.presets_dir() / voice / f"{lang}.codes.npy").exists():
                continue
            entry = res.setdefault(key, {"voice": voice, "lang": lang, "attempts": []})
            if any(a["seed_offset"] == args.seed_offset for a in entry["attempts"]):
                continue
            if args.only_failed and (not entry["attempts"] or max(
                    a.get("ratio", 1.0) for a in entry["attempts"]) >= THRESHOLD):
                continue
            t0 = time.perf_counter()
            audio = eng.synth(TEST_TEXT[lang], f"higgs:{voice}", 1.0, lang,
                              seed_offset=args.seed_offset)
            took = time.perf_counter() - t0
            wav = work / "reads" / f"{voice}_{lang}_s{args.seed_offset}.wav"
            write_wav(wav, audio, 44100)
            entry["attempts"].append({"seed_offset": args.seed_offset, "wav": str(wav),
                                      "text": TEST_TEXT[lang], "seconds": len(audio) / 44100,
                                      "rtf": took / max(len(audio) / 44100, 1e-6)})
            save_results(work, res)
            print(f"{key} s{args.seed_offset}: {len(audio) / 44100:.1f} s, RTF "
                  f"{entry['attempts'][-1]['rtf']:.2f}", flush=True)


def norm_words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def cmd_asr(args) -> None:
    import numpy as np
    import soxr
    from faster_whisper import WhisperModel

    work = Path(args.work)
    res = load_results(work)
    model = WhisperModel(args.whisper, device="cpu", compute_type="int8")
    for key, entry in res.items():
        for att in entry["attempts"]:
            if "ratio" in att:
                continue
            with wave.open(att["wav"], "rb") as wf:
                rate = wf.getframerate()
                pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
            x = soxr.resample(pcm.astype(np.float32) / 32768.0, rate, 16000)
            segs, _ = model.transcribe(x, language=entry["lang"], beam_size=5)
            heard = " ".join(s.text.strip() for s in segs)
            att["asr"] = heard
            att["ratio"] = difflib.SequenceMatcher(
                None, norm_words(att["text"]), norm_words(heard)).ratio()
            save_results(work, res)
            print(f"{key} s{att['seed_offset']}: {att['ratio']:.2f}  {heard}", flush=True)


def best_ratio(entry: dict) -> float | None:
    vals = [a["ratio"] for a in entry["attempts"] if "ratio" in a]
    return max(vals) if vals else None


def cmd_prune(args) -> None:
    from vibetts import higgs

    work = Path(args.work)
    res = load_results(work)
    for key, entry in res.items():
        best = best_ratio(entry)
        tried = len([a for a in entry["attempts"] if "ratio" in a])
        if best is None or best >= THRESHOLD or tried < 2 or entry.get("dropped"):
            continue
        folder = higgs.presets_dir() / entry["voice"]
        for suffix in (".codes.npy", ".txt"):
            (folder / f"{entry['lang']}{suffix}").unlink(missing_ok=True)
        entry["dropped"] = True
        print(f"dropped {key} (best ratio {best:.2f})", flush=True)
    save_results(work, res)


def cmd_table(args) -> None:
    from vibetts import higgs

    res = load_results(Path(args.work))
    langs = languages(None)
    lines = ["| lang | " + " | ".join(args.voices) + " |", "|---|" + "---|" * len(args.voices)]
    for lang in langs:
        cells = []
        for voice in args.voices:
            entry = res.get(f"{voice}/{lang}")
            exists = (higgs.presets_dir() / voice / f"{lang}.codes.npy").exists()
            if entry is None or best_ratio(entry) is None:
                cells.append("not checked" if exists else "missing")
            else:
                state = "dropped" if entry.get("dropped") or not exists else "kept"
                tries = len(entry["attempts"])
                cells.append(f"{best_ratio(entry):.2f} {state}" + (f" ({tries} takes)" if tries > 1 else ""))
        lines.append(f"| {lang} | " + " | ".join(cells) + " |")
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("step", choices=["make", "check", "asr", "prune", "table"])
    ap.add_argument("--voices", nargs="+", default=list(VOICES))
    ap.add_argument("--langs", nargs="+", default=None, help="default: all (priority order)")
    ap.add_argument("--work", default=str(ROOT / ".claude" / "temp" / "vtf09a" / "check"))
    ap.add_argument("--higgs-dir", default=str(ROOT / "models" / "higgs-audio-v3"))
    ap.add_argument("--voices-dir", default=str(ROOT / "models" / "higgs-voices"))
    ap.add_argument("--supertonic-dir", default=str(ROOT / "models" / "supertonic-3"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--only-failed", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--whisper", default="medium")
    args = ap.parse_args(argv)
    {"make": cmd_make, "check": cmd_check, "asr": cmd_asr, "prune": cmd_prune,
     "table": cmd_table}[args.step](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
