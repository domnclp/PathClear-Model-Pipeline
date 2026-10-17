"""Build the comparison summary PDF from results.csv, the saved evaluation reports, and the figures.

Every number comes from committed or artifact files, so rerunning reproduces the report.
Run with the repository .venv (reportlab).
"""
import csv
import json
from pathlib import Path
import subprocess

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
FIGURES = HERE / "figures"
OUTPUT = HERE / "PathClear_Detector_Comparison_Report.pdf"
ORDER = ("M0", "M2", "M1", "M3", "M4")
NAMES = {"M0": "FOMO v2", "M1": "ESPDet-Pico", "M2": "YOLO-FastestV2", "M3": "YOLO11n", "M4": "SSDLite-MobileNetV2"}
REPORTS = {"M0": "m0-fomo-evaluation", "M1": "m1-espdet-evaluation-1200", "M2": "m2-yolofastest-evaluation",
           "M3": "m3-yolo11n-evaluation", "M4": "m4-ssdlite-evaluation"}
INK, MUTED, RULE, HEAD = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#c9c8c3"), colors.HexColor("#eceae4")


def fonts():
    base = Path("C:/Windows/Fonts")
    for name, file in (("TNR", "times.ttf"), ("TNR-Bold", "timesbd.ttf"), ("TNR-Italic", "timesi.ttf"),
                       ("TNR-BoldItalic", "timesbi.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(base / file)))
    pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-Bold", italic="TNR-Italic", boldItalic="TNR-BoldItalic")


def styles():
    body = ParagraphStyle("body", fontName="TNR", fontSize=12, leading=16, textColor=INK, spaceAfter=8)
    return {"title": ParagraphStyle("title", parent=body, fontName="TNR-Bold", fontSize=17, leading=21,
                                    alignment=TA_CENTER, spaceAfter=6),
            "subtitle": ParagraphStyle("subtitle", parent=body, alignment=TA_CENTER, textColor=MUTED, fontSize=11),
            "h1": ParagraphStyle("h1", parent=body, fontName="TNR-Bold", fontSize=14, leading=18, spaceBefore=10,
                                 spaceAfter=6),
            "body": body,
            "bullet": ParagraphStyle("bullet", parent=body, leftIndent=18, bulletIndent=6, spaceAfter=4),
            "cell": ParagraphStyle("cell", parent=body, fontSize=9.5, leading=11.5, spaceAfter=0),
            "cellb": ParagraphStyle("cellb", parent=body, fontName="TNR-Bold", fontSize=9.5, leading=11.5, spaceAfter=0),
            "caption": ParagraphStyle("caption", parent=body, fontSize=10, leading=13, textColor=MUTED,
                                      fontName="TNR-Italic", spaceBefore=4)}


def table(rows, widths, s, header=True):
    data = [[Paragraph(str(c), s["cellb"] if (header and r == 0) or c_i == 0 else s["cell"]) for c_i, c in enumerate(row)]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
                           ("BACKGROUND", (0, 0), (-1, 0), HEAD if header else colors.white),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return t


def rotated(path):
    """Landscape figure turned 90 degrees so it can fill a portrait page, as in a thesis."""
    from PIL import Image as PILImage
    target = ART / "report-build" / (path.stem + "_rotated.png")  # Build intermediate, kept out of figures/.
    target.parent.mkdir(exist_ok=True)
    with PILImage.open(path) as im:
        im.rotate(90, expand=True).save(target)
    return target


def figure(path, width, caption, s):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    return KeepTogether([Image(str(path), width=width, height=width * h / w), Paragraph(caption, s["caption"])])


def main():
    fonts()
    s = styles()
    rows = {r["model_id"]: r for r in csv.DictReader((HERE / "results.csv").open(encoding="utf-8"))}
    reports = {m: json.loads((ART / folder / "report.json").read_text(encoding="utf-8")) for m, folder in REPORTS.items()}
    sens = json.loads((ART / "threshold-sensitivity-final" / "sensitivity.json").read_text(encoding="utf-8"))["models"]
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
    base = {k: float(rows["M0"][k]) for k in ("macs", "int8_bytes", "params")}
    pct = lambda v: f"{float(v) * 100:.1f}%"  # noqa: E731

    def size_recall(m, key):
        return reports[m]["selected"]["coarse_size_recall"][key]["recall"]

    def micro(m, k):
        return reports[m]["selected"]["micro"][k]

    def per_class(m, c):
        return reports[m]["selected"]["per_class"][c]["recall"]

    story = [Paragraph("PathClear Standalone Detector Comparison", s["title"]),
             Paragraph("FOMO v2 versus four lightweight detectors for the ESP32-CAM no-phone fallback mode", s["subtitle"]),
             Paragraph(f"Results summary, 17 October 2026 &nbsp;|&nbsp; branch <i>fomo-model-comparison</i>, commit {commit} "
                       "&nbsp;|&nbsp; desktop phases complete; on-board phases pending", s["subtitle"]),
             Spacer(1, 10)]

    story += [Paragraph("1. Summary", s["h1"]),
              Paragraph("Five INT8 object detectors were trained or reused on the same Cityscapes Person/Vehicle data and "
                        "scored with one shared centroid scorer and one pre-registered operating-point rule. FOMO v2 has the "
                        "lowest F1 of the five (49.8%) but by far the lowest cost: every alternative needs at least 1.3 times "
                        "its computation and a 5.9 times larger model file. Accuracy rises with cost across the candidates, "
                        "so whether FOMO is the best choice for PathClear depends on the ESP32-CAM flash, memory, and frame-rate "
                        "gates, which still require on-board measurement.", s["body"])]
    key = [f"<b>Most accurate:</b> SSDLite-MobileNetV2, F1 {pct(rows['M4']['f1'])} at "
           f"{float(rows['M4']['macs']) / base['macs']:.1f}x FOMO's compute; YOLO11n reaches "
           f"{pct(sens['M3']['extended_0.05_0.95']['f1'])} when the threshold sweep is extended, at "
           f"{float(rows['M3']['macs']) / base['macs']:.1f}x the compute and a 2.8 MB model.",
           f"<b>Closest challenger:</b> YOLO-FastestV2, F1 {pct(rows['M2']['f1'])} (3.6 points above FOMO) at only "
           f"{float(rows['M2']['macs']) / base['macs']:.1f}x the compute, but with poorly localized boxes "
           f"(mAP@0.5 {reports['M2']['map_0_5']['mean']:.3f}) and no license file.",
           f"<b>Espressif's ESPDet-Pico:</b> F1 {pct(rows['M1']['f1'])} after its official 1,200-epoch recipe "
           f"({pct(sens['M1']['extended_0.05_0.95']['f1'])} with the extended sweep), at "
           f"{float(rows['M1']['macs']) / base['macs']:.1f}x FOMO's compute.",
           f"<b>FOMO v2:</b> F1 {pct(rows['M0']['f1'])}, 16.3 M MACs per frame, and a 70,736-byte INT8 model, the "
           "smallest and cheapest candidate by a wide margin.",
           "<b>Defensible claim today:</b> FOMO is the most resource-efficient candidate, not the most accurate one."]
    story += [Paragraph(k, s["bullet"], bulletText="\u2022") for k in key]

    story += [Paragraph("2. Method", s["h1"]),
              Paragraph("The comparison targets PathClear's fallback mode, in which FOMO alone runs on the AI-Thinker "
                        "ESP32-CAM when no phone is connected; in the primary phone-connected mode, FOMO and Fast-SCNN run on "
                        "the phone. The decision rule was committed before any candidate result existed.", s["body"])]
    method = [["Control", "Applied to every candidate"],
              ["Data", "Cityscapes rebuilt on this laptop and verified to reproduce FOMO v2's exact recorded counts "
                       "(3,860 TP, 1,509 FP, 6,259 FN, 1,072 ignored); 2,975 train / 500 validation images, Person and Vehicle"],
              ["Input", "160 x 160 RGB, FOMO v2's direct bilinear stretch, identical cached pixels for every trainer"],
              ["Quantization", "Full INT8; the same 100 training calibration images, checked by image ID"],
              ["Scoring", "FOMO v2's centroid matching: same class, one-to-one, within 8 input pixels; void/crowd "
                          "regions ignored. Box models use the box center; mAP@0.5 is a secondary column"],
              ["Operating point", "Highest recall among thresholds 0.20-0.95 with precision of at least 70% (the v2 rule)"],
              ["Training", "Each model's official recipe, with budgets and stopping rules pre-registered in PLAN.md"],
              ["Hard gates", "G1 runtime support, G2 flash, G3 RAM, G4 at least 3 FPS on the board (3 FPS pending team sign-off)"]]
    story.append(table(method, [1.3 * inch, 5.2 * inch], s))

    story += [Paragraph("3. Results", s["h1"]),
              Paragraph("All figures are Cityscapes validation results from INT8 models. The validation split also selected "
                        "checkpoints and thresholds, so none of these is an independent test.", s["body"])]
    header = [""] + [f"{m}<br/>{NAMES[m]}" for m in ORDER]
    result_rows = [
        ["Selected threshold"] + [f"{reports[m]['selected']['threshold']:.2f}" for m in ORDER],
        ["Precision"] + [pct(micro(m, "precision")) for m in ORDER],
        ["Recall"] + [pct(micro(m, "recall")) for m in ORDER],
        ["<b>F1 (pre-registered)</b>"] + [f"<b>{pct(micro(m, 'f1'))}</b>" for m in ORDER],
        ["F1, sweep to 0.05"] + [pct(sens[m]["extended_0.05_0.95"]["f1"]) for m in ORDER],
        ["Person recall"] + [pct(per_class(m, "person")) for m in ORDER],
        ["Vehicle recall"] + [pct(per_class(m, "vehicle")) for m in ORDER],
        ["Recall &lt; 2 px"] + [pct(size_recall(m, "lt2")) for m in ORDER],
        ["Recall 2-8 px"] + [pct(size_recall(m, "2to8")) for m in ORDER],
        ["Recall &gt;= 8 px"] + [pct(size_recall(m, "ge8")) for m in ORDER],
        ["mAP@0.5"] + ["-" if m == "M0" else f"{reports[m]['map_0_5']['mean']:.3f}" for m in ORDER],
        ["Parameters"] + [f"{int(rows[m]['params']):,}" for m in ORDER],
        ["MACs (x FOMO)"] + [f"{float(rows[m]['macs']) / 1e6:.1f} M ({float(rows[m]['macs']) / base['macs']:.1f}x)" for m in ORDER],
        ["INT8 file (x FOMO)"] + [f"{int(rows[m]['int8_bytes']) / 1000:,.0f} KB ({int(rows[m]['int8_bytes']) / base['int8_bytes']:.1f}x)" for m in ORDER],
        ["Runtime"] + ["TFLite Micro", "ESP-DL", "ESP-DL", "ESP-DL", "TFLite Micro"],
    ]
    story.append(table([header] + result_rows, [1.35 * inch] + [1.03 * inch] * 5, s))
    story += [Spacer(1, 10),
              figure(FIGURES / "accuracy_vs_cost.png", 5.6 * inch,
                     "Figure 1. F1 against computation per frame. Filled markers use the pre-registered rule; hollow markers "
                     "show the extended-sweep sensitivity check. Upper left is better.", s)]
    story += [Paragraph("4. Key findings", s["h1"])]
    findings = [
        "FOMO v2 is not the most accurate detector. Every alternative scored a higher F1 on identical data, so the thesis "
        "should not claim accuracy superiority.",
        "Accuracy and cost rise together. No candidate beats FOMO without at least 1.3x its computation; the larger gains "
        "(SSDLite +12.3 F1 points, YOLO11n up to +17.0 with the extended sweep) cost 2.2-12.5x the computation.",
        "YOLO-FastestV2 is the closest challenger at 1.3x the compute and 5.9x the model size, but its boxes are poorly "
        "placed (mAP@0.5 0.026) and its repository has no license, which rules it out for a commercial product.",
        "Training budget mattered for from-scratch models: ESPDet-Pico improved from 35.5% F1 and 3.8% person recall at a "
        "300-epoch cap to 52.2% and 30.0% after the official 1,200 epochs, as the pre-registered rule required.",
        "Small objects are hard for every model: at 160 x 160 most pedestrians are under 8 pixels wide, and no model finds "
        "more than about a third of objects under 2 pixels. This is a camera-resolution limit, not a FOMO weakness.",
        "Peak working memory follows the input resolution and early layers, not model size: FOMO and SSDLite share the "
        "largest buffer (384 KB), and every model needs the 4 MB PSRAM."]
    story += [Paragraph(f, s["bullet"], bulletText=f"{i}.") for i, f in enumerate(findings, 1)]

    story += [PageBreak(),
              figure(rotated(FIGURES / "comparison.png"), 2.8 * inch,
                     "Figure 2. (a) Precision, recall and F1; (b) recall by object size in the 160 x 160 input; "
                     "(c) cost relative to FOMO v2 on a log scale. ESP-DL models run in the ESP-PPQ simulator, so they have "
                     "no desktop latency comparable to TFLite.", s)]

    story += [Paragraph("5. Hardware gate status", s["h1"])]
    gates = [["Gate", "Status", "Evidence"],
             ["G1 runtime support", "Static pass, all five", "Every operator is on TFLite Micro's or ESP-DL's supported list; "
              "the classic ESP32 runs ESP-DL as C kernels without vector instructions"],
             ["G2 flash", "Pending", "Model files range from 71 KB to 2.8 MB within 4 MB of shared flash; firmware builds need "
              "ESP-IDF (not installed: 12 GB free on drive C:). YOLO11n is a likely failure"],
             ["G3 RAM", "Estimate only", "Peak INT8 activations 205-384 KB for all models; all exceed free internal SRAM and fit "
              "in the 4 MB PSRAM, so this does not separate the candidates"],
             ["G4 at least 3 FPS", "Pending", "Requires a working ESP32-CAM. Espressif reports YOLO11n at 1,335 ms per frame at "
              "320 x 320 even on the faster ESP32-S3, so it is unlikely to pass"]]
    story.append(KeepTogether([table(gates, [1.3 * inch, 1.25 * inch, 3.95 * inch], s)]))

    story += [Paragraph("6. Deviations and limitations", s["h1"])]
    limits = [
        "Cityscapes is German street data; Philippine conditions, jeepneys, and ESP32-CAM optics are not represented.",
        "Validation data chose checkpoints and thresholds for every model; recorded ESP32-CAM walking footage is the "
        "planned independent test.",
        "Three models selected the 0.20 sweep floor; the extended-sweep check is reported alongside, not substituted.",
        "Starting points differ by each model's standard recipe: FOMO and SSDLite use ImageNet weights, YOLO11n COCO "
        "detection weights, YOLO-FastestV2 an ImageNet backbone, ESPDet-Pico none.",
        "YOLO11n's best epoch was 99 of its official 100, so it may be slightly under-trained.",
        "ESP-DL models were scored in the ESP-PPQ simulator, a different INT8 engine from TFLite; their exports differ by "
        "tens of bytes between runs.",
        "Compatibility shims kept third-party code unmodified under newer libraries; they are documented in README.md.",
        "Licenses: ESPDet-Pico and YOLO11n are AGPL-3.0 (Ultralytics); YOLO-FastestV2 has no license file."]
    story += [Paragraph(f, s["bullet"], bulletText="\u2022") for f in limits]

    story += [Paragraph("7. Next steps and open decisions", s["h1"])]
    steps = [
        "<b>Team / Ms. Capule:</b> confirm the 3 FPS G4 threshold and review the AGPL and no-license findings for "
        "Commercialization.",
        "<b>ESP-IDF:</b> free disk space (about 4-5 GB needed) and install ESP-IDF to build per-model firmware for G2.",
        "<b>Phase 4:</b> with a working ESP32-CAM, measure flash, RAM, and frame rate for each model; FOMO versus "
        "YOLO-FastestV2 and SSDLite is the decisive comparison.",
        "<b>Phase 5:</b> record and label ESP32-CAM walking footage for an independent test of the gate-passing models.",
        "<b>Phase 6:</b> apply the gates and F1 ranking in the decision matrix and write the Chapter 3 paragraph."]
    story += [Paragraph(f, s["bullet"], bulletText="\u2022") for f in steps]
    story += [Spacer(1, 6), Paragraph("Sources: tinyml/fomo/comparison/README.md, PLAN.md, results.csv, and the evaluation "
                                      "reports under tinyml/fomo/artifacts/.", s["caption"])]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("TNR", 9)
        canvas.setFillColor(MUTED)
        canvas.drawString(inch, 0.6 * inch, "PathClear detector comparison - desktop results, not on-device measurements")
        canvas.drawRightString(letter[0] - inch, 0.6 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUTPUT), pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=inch,
                            bottomMargin=inch, title="PathClear Standalone Detector Comparison",
                            author="PathClear thesis team", subject="FOMO v2 versus four lightweight detectors")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
