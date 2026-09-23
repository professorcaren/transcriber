// Transcript export formats. `doc` is built by app.js:
// { file, duration, created, diarized, transcribed, models: {asr, diar}, names: [..8], turns: [{speaker, start, end, text, words, edited}], segs }

const pad = (n, w = 2) => String(n).padStart(w, '0');
export function hms(t, msSep = null) {
  t = Math.max(0, t);
  const s = `${pad(Math.floor(t / 3600))}:${pad(Math.floor(t / 60) % 60)}:${pad(Math.floor(t % 60))}`;
  return msSep ? s + msSep + pad(Math.round((t % 1) * 1000) % 1000, 3) : s;
}
const who = (doc, t) => (doc.diarized && t.speaker != null ? doc.names[t.speaker] : null);
const credits = doc => [
  doc.models.asr && `${doc.models.asr} (words)`,
  doc.models.diar && `${doc.models.diar} (speakers)`,
].filter(Boolean).join(' and ');

// ---------- reading formats ----------
export function toTxt(doc) {
  const lines = [
    doc.file,
    `Duration: ${hms(doc.duration)}`,
    `Transcribed ${doc.created.toLocaleDateString()} with ${credits(doc)}.`,
    'Automatic transcript: check names, numbers and unclear passages against the audio.',
    '', '',
  ];
  for (const t of doc.turns) {
    const name = who(doc, t);
    if (!doc.transcribed) { lines.push(`[${hms(t.start)} - ${hms(t.end)}] ${name}`); continue; }
    lines.push(`[${hms(t.start)}] ${name ? name + ': ' : ''}${t.text}`, '');
  }
  return lines.join('\n');
}

const xml = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const run = (text, props = '') => `<w:r>${props ? `<w:rPr>${props}</w:rPr>` : ''}<w:t xml:space="preserve">${xml(text)}</w:t></w:r>`;
const para = (runs, pPr = '') => `<w:p>${pPr ? `<w:pPr>${pPr}</w:pPr>` : ''}${runs}</w:p>`;

export function toDocx(doc) {
  const body = [
    para(run(doc.file, '<w:b/><w:sz w:val="32"/>'), '<w:spacing w:after="80"/>'),
    para(run(`Duration ${hms(doc.duration)} · transcribed ${doc.created.toLocaleDateString()} with ${credits(doc)}.`, '<w:color w:val="666666"/><w:sz w:val="18"/>')),
    para(run('Automatic transcript: check names, numbers and unclear passages against the audio.', '<w:i/><w:color w:val="666666"/><w:sz w:val="18"/>'), '<w:spacing w:after="360"/>'),
  ];
  for (const t of doc.turns) {
    const name = who(doc, t);
    const head = run(`[${hms(t.start)}] `, '<w:color w:val="808080"/>') + (name ? run(name + ': ', '<w:b/>') : '');
    if (!doc.transcribed) { body.push(para(head + run(`until ${hms(t.end)}`, '<w:color w:val="808080"/>'))); continue; }
    const paras = t.text.split(/\n\s*\n/).map(s => s.trim()).filter(Boolean);
    paras.forEach((p, i) => body.push(para((i === 0 ? head : '') + run(p), '<w:spacing w:after="160"/>')));
  }
  const files = {
    '[Content_Types].xml': `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>`,
    '_rels/.rels': `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>`,
    'word/_rels/document.xml.rels': `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>`,
    'word/styles.xml': `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/><w:sz w:val="22"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults></w:styles>`,
    'word/document.xml': `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>${body.join('')}<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr></w:body></w:document>`,
  };
  return zip(files);
}

// ---------- subtitles ----------
function wrap(text, width = 42) {
  if (text.length <= width) return text;
  const mid = text.lastIndexOf(' ', Math.min(text.length - 1, Math.ceil(text.length / 2) + 6));
  return mid > 0 ? text.slice(0, mid) + '\n' + text.slice(mid + 1) : text;
}

// cues of at most ~84 characters / 6 seconds, from word timings when the turn is unedited,
// otherwise by spreading the edited sentences over the turn's time span
function cues(doc) {
  const out = [];
  for (const t of doc.turns) {
    if (!t.text) continue;
    const name = who(doc, t);
    const parts = [];
    if (!t.edited && t.words?.length) {
      let cur = null;
      for (const w of t.words) {
        const txt = w.text.trim();
        if (cur && (cur.text.length + txt.length > 84 || w.end - cur.start > 6)) { parts.push(cur); cur = null; }
        if (!cur) cur = { start: w.start, end: w.end, text: txt };
        else { cur.text += ' ' + txt; cur.end = w.end; }
        if (/[.?!]$/.test(txt) && cur.text.length > 30) { parts.push(cur); cur = null; }
      }
      if (cur) parts.push(cur);
    } else {
      const sents = t.text.replace(/\s+/g, ' ').match(/[^.?!]+[.?!]*\s*/g) || [t.text];
      const total = sents.reduce((a, s) => a + s.length, 0), span = t.end - t.start;
      let at = t.start;
      for (const s of sents) { const d = span * s.length / total; parts.push({ start: at, end: at + d, text: s.trim() }); at += d; }
    }
    parts.forEach((p, i) => out.push({ ...p, end: Math.max(p.end, p.start + 0.5), name, first: i === 0 }));
  }
  return out;
}

export function toSrt(doc) {
  return cues(doc).map((c, i) =>
    `${i + 1}\n${hms(c.start, ',')} --> ${hms(c.end, ',')}\n${wrap((c.name && c.first ? `[${c.name}] ` : '') + c.text)}\n`).join('\n');
}

export function toVtt(doc) {
  return 'WEBVTT\n\n' + cues(doc).map(c =>
    `${hms(c.start, '.')} --> ${hms(c.end, '.')}\n${c.name ? `<v ${c.name}>` : ''}${wrap(c.text)}\n`).join('\n');
}

// ---------- analysis formats ----------
const csvCell = v => (v == null ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
export function toCsv(doc) {
  const rows = [['start', 'end', 'start_seconds', 'end_seconds', 'speaker', 'text']];
  for (const t of doc.turns) rows.push([hms(t.start), hms(t.end), t.start.toFixed(2), t.end.toFixed(2), who(doc, t) ?? '', t.text.replace(/\s*\n\s*\n\s*/g, ' ')]);
  return '﻿' + rows.map(r => r.map(csvCell).join(',')).join('\r\n') + '\r\n';
}

export function toJson(doc) {
  const r2 = x => Math.round(x * 100) / 100;
  return JSON.stringify({
    file: doc.file, duration_seconds: r2(doc.duration), created: doc.created.toISOString(), models: doc.models,
    speakers: doc.diarized ? Object.fromEntries([...new Set(doc.turns.map(t => t.speaker))].map(s => [s, doc.names[s]])) : null,
    turns: doc.turns.map(t => ({ start: r2(t.start), end: r2(t.end), speaker: t.speaker, speaker_name: who(doc, t), text: t.text, edited: !!t.edited })),
    words: doc.turns.flatMap(t => (t.words || []).map(w => ({ text: w.text.trim(), start: r2(w.start), end: r2(w.end), speaker: t.speaker }))),
    diarization_segments: doc.segs?.map(s => ({ start: s.start, end: s.end, speaker: s.speaker })) ?? null,
  }, null, 1);
}

export function toRttm(doc) {
  const id = doc.file.replace(/\.[^.]+$/, '').replace(/\s+/g, '_');
  return doc.segs.map(s =>
    `SPEAKER ${id} 1 ${s.start.toFixed(2)} ${(s.end - s.start).toFixed(2)} <NA> <NA> ${doc.names[s.speaker].replace(/\s+/g, '_')} <NA> <NA>`).join('\n') + '\n';
}

// ---------- minimal ZIP (stored, no compression) for .docx ----------
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(b) { let c = 0xffffffff; for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }

function zip(files) {
  const enc = new TextEncoder(), local = [], central = [];
  let offset = 0;
  const d = new Date();
  const dosTime = (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
  const dosDate = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
  for (const [name, content] of Object.entries(files)) {
    const nameB = enc.encode(name), data = enc.encode(content), crc = crc32(data);
    const h = new DataView(new ArrayBuffer(30));
    h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true); h.setUint16(8, 0, true);
    h.setUint16(10, dosTime, true); h.setUint16(12, dosDate, true); h.setUint32(14, crc, true);
    h.setUint32(18, data.length, true); h.setUint32(22, data.length, true); h.setUint16(26, nameB.length, true); h.setUint16(28, 0, true);
    local.push(new Uint8Array(h.buffer), nameB, data);
    const c = new DataView(new ArrayBuffer(46));
    c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true); c.setUint16(10, 0, true);
    c.setUint16(12, dosTime, true); c.setUint16(14, dosDate, true); c.setUint32(16, crc, true);
    c.setUint32(20, data.length, true); c.setUint32(24, data.length, true); c.setUint16(28, nameB.length, true);
    c.setUint32(42, offset, true);
    central.push(new Uint8Array(c.buffer), nameB);
    offset += 30 + nameB.length + data.length;
  }
  const cSize = central.reduce((a, b) => a + b.length, 0);
  const e = new DataView(new ArrayBuffer(22));
  e.setUint32(0, 0x06054b50, true); e.setUint16(8, Object.keys(files).length, true); e.setUint16(10, Object.keys(files).length, true);
  e.setUint32(12, cSize, true); e.setUint32(16, offset, true);
  return new Blob([...local, ...central, new Uint8Array(e.buffer)], { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' });
}
