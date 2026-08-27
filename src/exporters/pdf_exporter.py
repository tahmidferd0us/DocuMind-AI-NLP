import io
from typing import Dict, Any, List, Optional
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

def generate_pdf_report(
    document_info: Dict[str, Any],
    extractive_summary: Optional[Dict[str, Any]] = None,
    abstractive_summary: Optional[Dict[str, Any]] = None,
    entities_data: Optional[Dict[str, Any]] = None,
    keywords_data: Optional[List[Dict[str, Any]]] = None,
    analytics_data: Optional[Dict[str, Any]] = None,
    qa_history: Optional[List[Dict[str, Any]]] = None
) -> bytes:
    """Generates a professional ReportLab PDF report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E3A8A"),
        alignment=1, # Center
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B"),
        alignment=1,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#334155"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=body_style,
        leftIndent=12,
        spaceAfter=4
    )

    story = []
    
    # Title & Subtitle
    story.append(Paragraph("DocuMind AI — Document Intelligence Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')} | Smart NLP Platform (KOI)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=12))

    # 1. Metadata Table
    story.append(Paragraph("1. Document Metadata & Reading Analytics", h1_style))
    
    meta_data = [
        [Paragraph("<b>File Name</b>", body_style), Paragraph(str(document_info.get("filename", "Unknown")), body_style)],
        [Paragraph("<b>Page Count</b>", body_style), Paragraph(str(document_info.get("page_count", 1)), body_style)],
        [Paragraph("<b>Total Word Count</b>", body_style), Paragraph(f"{analytics_data.get('word_count', document_info.get('total_words', 0)):,} words" if analytics_data else "N/A", body_style)],
        [Paragraph("<b>Reading Time</b>", body_style), Paragraph(f"{analytics_data.get('reading_time_min', 0)} min (Speaking: {analytics_data.get('speaking_time_min', 0)} min)" if analytics_data else "N/A", body_style)],
        [Paragraph("<b>Readability</b>", body_style), Paragraph(f"{analytics_data.get('readability', {}).get('score', 'N/A')} — {analytics_data.get('readability', {}).get('level', '')} ({analytics_data.get('readability', {}).get('grade', '')})" if analytics_data else "N/A", body_style)]
    ]
    
    t = Table(meta_data, colWidths=[150, 350])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F8FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # 2. Summaries
    story.append(Paragraph("2. Dual Document Summaries", h1_style))
    
    if abstractive_summary and abstractive_summary.get("summary"):
        story.append(Paragraph("A. Abstractive Neural Summary (Synthesis)", h2_style))
        story.append(Paragraph(abstractive_summary["summary"].replace("\n", "<br/>"), body_style))
        story.append(Spacer(1, 6))
        
    if extractive_summary and extractive_summary.get("sentences"):
        story.append(Paragraph(f"B. Extractive Summary (LexRank Top {len(extractive_summary['sentences'])} Sentences)", h2_style))
        for s in extractive_summary["sentences"]:
            story.append(Paragraph(f"• {s}", bullet_style))
        story.append(Spacer(1, 8))

    # 3. Topics & Entities
    story.append(Paragraph("3. Key Topics & Named Entities", h1_style))
    if keywords_data:
        story.append(Paragraph("Top Key Phrases (KeyBERT)", h2_style))
        kw_str = ", ".join([f"{k['keyword']} ({k['score']})" for k in keywords_data[:10]])
        story.append(Paragraph(kw_str, body_style))
        
    if entities_data and entities_data.get("entities_by_type"):
        story.append(Paragraph("Extracted Entities (spaCy NER)", h2_style))
        for grp, items in entities_data["entities_by_type"].items():
            if items:
                n_str = ", ".join([f"{it['name']} (x{it['count']})" for it in items[:6]])
                story.append(Paragraph(f"<b>{grp}:</b> {n_str}", bullet_style))
        story.append(Spacer(1, 8))

    # 4. Q&A
    if qa_history:
        story.append(Paragraph("4. Grounded Question Answering History", h1_style))
        for idx, qa in enumerate(qa_history, start=1):
            story.append(Paragraph(f"<b>Q{idx}: {qa.get('question', '')}</b>", body_style))
            story.append(Paragraph(f"Answer: {qa.get('answer', '')}", bullet_style))
            if qa.get("sources"):
                citations = list(set([f"Page {s.get('page')}" for s in qa.get("sources", [])]))
                story.append(Paragraph(f"<i>Citations: {', '.join(citations)}</i>", ParagraphStyle('Cit', parent=bullet_style, textColor=colors.HexColor("#64748B"))))
            story.append(Spacer(1, 4))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
