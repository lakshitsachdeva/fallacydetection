// Build an IEEE-style Word version of the paper from paper.json (made by tex_to_blocks.py).
// Usage: node build_docx.js  -> ../fallacylens_ieee.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  WidthType, BorderStyle, SectionType, LevelFormat, Footer, PageNumber, TabStopType,
} = require("docx");

const paper = JSON.parse(fs.readFileSync(path.join(__dirname, "paper.json"), "utf8"));
const FONT = "Times New Roman";
const COL_W = 5040; // 3.5in column (DXA)
const FULL_W = 10440; // 7.25in text width
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"];

// Inline markup from the converter: **bold**, //italic//, ``code``
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|(?<!:)\/\/.+?\/\/|``[^`]+``)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), ...base, bold: true }));
    else if (t.startsWith("//")) out.push(new TextRun({ text: t.slice(2, -2), ...base, italics: !base.italics }));
    else out.push(new TextRun({ text: t.slice(2, -2), ...base, font: "Courier New" }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const body = (text) =>
  new Paragraph({ children: runs(text), alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 202 }, spacing: { after: 0, line: 240 } });

function figure(b) {
  const { w, h } = pngSize(b.image);
  const maxW = (b.wide ? 6.4 : 3.45) * 96; // px at 96 dpi
  const width = Math.min(maxW, w);
  const height = Math.round((h * width) / w);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(b.image), transformation: { width, height } })],
    }),
    new Paragraph({
      alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 },
      children: [new TextRun({ text: `Fig. ${b.n}. `, size: 16 }), ...runs(b.caption, { size: 16 })],
    }),
  ];
}

function table(b) {
  const width = b.wide ? FULL_W : COL_W;
  const ncol = b.rows[0].length;
  // Column widths proportional to the longest cell in each column.
  const len = b.rows[0].map((_, j) => Math.max(...b.rows.map((r) => (r[j] || "").length)) + 2);
  const total = len.reduce((x, y) => x + y, 0);
  const widths = len.map((l) => Math.floor((width * l) / total));
  widths[ncol - 1] += width - widths.reduce((a, c) => a + c, 0);
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const rule = (sz) => ({ style: BorderStyle.SINGLE, size: sz, color: "000000" });
  const rows = b.rows.map((r, i) =>
    new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) =>
        new TableCell({
          width: { size: widths[j], type: WidthType.DXA },
          margins: { top: 20, bottom: 20, left: 60, right: 60 },
          borders: {
            top: i === 0 ? rule(8) : none,
            bottom: i === 0 ? rule(4) : i === b.rows.length - 1 ? rule(8) : none,
            left: none, right: none,
          },
          children: [new Paragraph({
            alignment: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
            children: runs(c, { size: 15 }),
          })],
        })),
    }));
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 160, after: 0 }, keepNext: true,
      children: [new TextRun({ text: `TABLE ${b.n}`, size: 16 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 80 }, keepNext: true,
      children: runs(b.caption, { size: 16, smallCaps: true }),
    }),
    new Table({ width: { size: width, type: WidthType.DXA }, columnWidths: widths, rows, alignment: AlignmentType.CENTER }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

// ---- assemble sections: title block (1 col), then 2-col body broken by 1-col wide floats
const page = { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1440, left: 900, right: 900 } };
const footer = new Footer({
  children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 16 })] })],
});
const sections = [];
let current = [];
const flush = (cols) => {
  if (!current.length) return;
  sections.push({
    properties: {
      page, type: sections.length ? SectionType.CONTINUOUS : undefined,
      column: cols === 2 ? { count: 2, space: 360, equalWidth: true } : { count: 1 },
    },
    footers: { default: footer },
    children: current,
  });
  current = [];
};

current.push(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: paper.title, size: 48 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: paper.authors, size: 22 })] }),
  ...paper.affil.map((a, i) =>
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: i === paper.affil.length - 1 ? 320 : 0 }, children: [new TextRun({ text: a, size: 20, italics: i === 0 })] })),
);
flush(1);

let sec = 0, sub = 0;
for (const b of paper.blocks) {
  if (b.type === "abstract") {
    current.push(new Paragraph({
      alignment: AlignmentType.JUSTIFIED, spacing: { after: 80 },
      children: [new TextRun({ text: "Abstract", bold: true, italics: true, size: 18 }), new TextRun({ text: "—", bold: true, size: 18 }), ...runs(b.text, { bold: true, size: 18 })],
    }));
  } else if (b.type === "keywords") {
    current.push(new Paragraph({
      alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 },
      children: [new TextRun({ text: "Index Terms", bold: true, italics: true, size: 18 }), new TextRun({ text: "—", bold: true, size: 18 }), ...runs(b.text, { bold: true, size: 18 })],
    }));
  } else if (b.type === "h1") {
    sub = 0;
    const label = b.numbered ? `${ROMAN[sec++]}. ${b.text}` : b.text;
    current.push(new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true,
      children: [new TextRun({ text: label, smallCaps: true, size: 20 })],
    }));
  } else if (b.type === "h2") {
    current.push(new Paragraph({
      spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new TextRun({ text: `${String.fromCharCode(65 + sub++)}. ${b.text}`, italics: true, size: 20 })],
    }));
  } else if (b.type === "p") {
    current.push(body(b.text));
  } else if (b.type === "list") {
    b.items.forEach((it) => current.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED, children: runs(it) })));
  } else if (b.type === "figure" || b.type === "table") {
    const parts = b.type === "figure" ? figure(b) : table(b);
    if (b.wide) {
      flush(2);
      current.push(...parts);
      flush(1);
    } else current.push(...parts);
  }
}

current.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 },
  children: [new TextRun({ text: "References", smallCaps: true, size: 20 })],
}));
paper.refs.forEach((r, i) => current.push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 40 },
  indent: { left: 360, hanging: 360 }, tabStops: [{ type: TabStopType.LEFT, position: 360 }],
  children: [new TextRun({ text: `[${i + 1}]\t`, size: 16 }), ...runs(r, { size: 16 })],
})));
flush(2);

const doc = new Document({
  creator: "FallacyLens authors",
  title: paper.title,
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 200 } } } }],
    }],
  },
  sections,
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "..", "fallacylens_ieee.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
