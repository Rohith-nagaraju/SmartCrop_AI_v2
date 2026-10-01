from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def build_report(result):
    b=BytesIO(); doc=SimpleDocTemplate(b); s=getSampleStyleSheet(); story=[Paragraph('SmartCrop AI Analysis Report',s['Title']),Spacer(1,12)]
    for k,v in [('Disease',result['disease']),('Confidence',f"{result['confidence']*100:.2f}%"),('Severity',f"{result['severity_percent']}% ({result['severity_class']})"),('DPI',result['dpi']),('Risk',f"{result['risk_score']} / 100 ({result['risk_level']})"),('Recommendation',result['recommendation'])]: story.append(Paragraph(f'<b>{k}</b>: {v}',s['BodyText'])); story.append(Spacer(1,7))
    doc.build(story); b.seek(0); return b
