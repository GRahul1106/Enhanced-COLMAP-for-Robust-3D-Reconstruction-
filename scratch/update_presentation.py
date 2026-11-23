"""
scratch/update_presentation.py

Systematically updates Quality_Guided_Enhancement_Final (1).pptx to produce
Quality_Guided_Enhancement_Final_Revised.pptx with all user requirements:
1. Adds verified actual before-and-after preview image to Slide 8.
2. Updates Slide 2 to clearly state selective enhancement across four datasets without claiming consistent improvement.
3. Updates Slide 7 table headers to 'Points Baseline' and 'Points Enh.' and adds an evidence-backed footnote.
4. Updates Slide 8 table with verified inliers and track length, preserving caveats.
5. Updates Slide 9 subtitle to 'Left: % Point Change | Right: Reprojection Error Change'.
6. Updates Slide 12 to remove bracket placeholders from dataset URL.
7. Preserves original presentation file intact.
"""

import os
import zipfile
import shutil
import xml.etree.ElementTree as ET

SOURCE_PPTX = r"E:\CV Project\Quality_Guided_Enhancement_Final (1).pptx"
TARGET_PPTX = r"E:\CV Project\Quality_Guided_Enhancement_Final_Revised.pptx"
PREVIEW_IMG = r"E:\CV Project\experiments\graham_hall\variant_e\previews\preview_IMG_0389.jpg"

NS = {
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
}

# Register namespaces to prevent ns0 / ns1 prefixes if re-serializing
for prefix, uri in [
    ('p', 'http://schemas.openxmlformats.org/presentationml/2006/main'),
    ('a', 'http://schemas.openxmlformats.org/drawingml/2006/main'),
    ('r', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'),
    ('c', 'http://schemas.openxmlformats.org/drawingml/2006/chart')
]:
    ET.register_namespace(prefix, uri)

def update_presentation():
    assert os.path.exists(SOURCE_PPTX), f"Source not found: {SOURCE_PPTX}"
    assert os.path.exists(PREVIEW_IMG), f"Preview image not found: {PREVIEW_IMG}"
    
    with zipfile.ZipFile(SOURCE_PPTX, 'r') as src_zip:
        files = {name: src_zip.read(name) for name in src_zip.namelist()}
        
    print(f"Loaded source PPTX with {len(files)} entries.")
    
    # -------------------------------------------------------------
    # 1. Update [Content_Types].xml
    # -------------------------------------------------------------
    ct_xml = files['[Content_Types].xml'].decode('utf-8')
    if 'Extension="jpg"' not in ct_xml:
        ct_tree = ET.fromstring(ct_xml)
        default_elem = ET.Element('{http://schemas.openxmlformats.org/package/2006/content-types}Default', {
            'Extension': 'jpg',
            'ContentType': 'image/jpeg'
        })
        ct_tree.append(default_elem)
        files['[Content_Types].xml'] = ET.tostring(ct_tree, encoding='utf-8', xml_declaration=True)
        print("Updated [Content_Types].xml with image/jpeg default extension.")
        
    # -------------------------------------------------------------
    # 2. Update Slide 2 (Motivation & Objectives)
    # -------------------------------------------------------------
    s2_tree = ET.fromstring(files['ppt/slides/slide2.xml'])
    for t_elem in s2_tree.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t'):
        if t_elem.text and 'Compare baseline vs. Variant E on four datasets' in t_elem.text:
            t_elem.text = 'Evaluate quality-guided selective enhancement (Variant E) before COLMAP across four datasets'
            print("Slide 2: Updated objective text.")
        elif t_elem.text and 'Evaluated honestly: four datasets, baseline vs. Variant E, same COLMAP protocol.' in t_elem.text:
            t_elem.text = 'Evaluated honestly: assesses sparse reconstruction across four datasets without assuming consistent improvement.'
            print("Slide 2: Updated bottom note text.")
    files['ppt/slides/slide2.xml'] = ET.tostring(s2_tree, encoding='utf-8', xml_declaration=True)
    
    # -------------------------------------------------------------
    # 3. Update Slide 7 (Table Headers & Evidence-Backed Footnote)
    # -------------------------------------------------------------
    s7_tree = ET.fromstring(files['ppt/slides/slide7.xml'])
    
    # Update table headers
    tbl = s7_tree.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}tbl')
    if tbl is not None:
        header_tr = tbl.findall('{http://schemas.openxmlformats.org/drawingml/2006/main}tr')[0]
        for tc in header_tr.findall('{http://schemas.openxmlformats.org/drawingml/2006/main}tc'):
            for t_elem in tc.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t'):
                if not t_elem.text: continue
                txt = t_elem.text.strip()
                if txt == 'Points base':
                    t_elem.text = 'Points Baseline'
                elif txt == 'Points E':
                    t_elem.text = 'Points Enh.'
                elif 'Reg. base' in txt:
                    t_elem.text = 'Reg. Base \u2192 Enh.'
                elif txt == 'Err base (px)':
                    t_elem.text = 'Err Base (px)'
                elif txt == 'Err E (px)':
                    t_elem.text = 'Err Enh. (px)'
        print("Slide 7: Updated table column headers.")
        
    # Update footnote banner shape
    for sp in s7_tree.findall('.//{http://schemas.openxmlformats.org/presentationml/2006/main}sp'):
        nv = sp.find('.//{http://schemas.openxmlformats.org/presentationml/2006/main}cNvPr')
        if nv is not None and nv.attrib.get('id') == '6':
            txBody = sp.find('{http://schemas.openxmlformats.org/presentationml/2006/main}txBody')
            if txBody is not None:
                # Clear existing paragraphs
                for p in list(txBody.findall('{http://schemas.openxmlformats.org/drawingml/2006/main}p')):
                    txBody.remove(p)
                
                # New paragraph 1: Footnote distinguishing Gerrard Hall (Phase 5) from South/Person/Graham (Variant E)
                p1 = ET.Element('{http://schemas.openxmlformats.org/drawingml/2006/main}p')
                pPr1 = ET.SubElement(p1, '{http://schemas.openxmlformats.org/drawingml/2006/main}pPr', {'marL': '0', 'indent': '0'})
                ET.SubElement(pPr1, '{http://schemas.openxmlformats.org/drawingml/2006/main}buNone')
                r1 = ET.SubElement(p1, '{http://schemas.openxmlformats.org/drawingml/2006/main}r')
                rPr1 = ET.SubElement(r1, '{http://schemas.openxmlformats.org/drawingml/2006/main}rPr', {
                    'lang': 'en-US', 'sz': '950', 'dirty': '0'
                })
                sf1 = ET.SubElement(rPr1, '{http://schemas.openxmlformats.org/drawingml/2006/main}solidFill')
                ET.SubElement(sf1, '{http://schemas.openxmlformats.org/drawingml/2006/main}srgbClr', {'val': '374151'})
                ET.SubElement(rPr1, '{http://schemas.openxmlformats.org/drawingml/2006/main}latin', {'typeface': 'Calibri'})
                t1 = ET.SubElement(r1, '{http://schemas.openxmlformats.org/drawingml/2006/main}t')
                t1.text = "*Footnote: Gerrard Hall evaluates the Phase 5 adaptive pipeline (blur < 80 unsharp mask, contrast < 60 CLAHE clip=2.0; 17 sharpened, 5 CLAHE, 1 both). South Building, Person Hall, and Graham Hall evaluate conservative Variant E (contrast < 52.0 & SIFT < 10k, CLAHE clip=1.2 only)."
                
                # New paragraph 2: Graham Hall subset notes and runtime caveat
                p2 = ET.Element('{http://schemas.openxmlformats.org/drawingml/2006/main}p')
                pPr2 = ET.SubElement(p2, '{http://schemas.openxmlformats.org/drawingml/2006/main}pPr', {'marL': '0', 'indent': '0'})
                ET.SubElement(pPr2, '{http://schemas.openxmlformats.org/drawingml/2006/main}buNone')
                r2 = ET.SubElement(p2, '{http://schemas.openxmlformats.org/drawingml/2006/main}r')
                rPr2 = ET.SubElement(r2, '{http://schemas.openxmlformats.org/drawingml/2006/main}rPr', {
                    'lang': 'en-US', 'sz': '950', 'dirty': '0'
                })
                sf2 = ET.SubElement(rPr2, '{http://schemas.openxmlformats.org/drawingml/2006/main}solidFill')
                ET.SubElement(sf2, '{http://schemas.openxmlformats.org/drawingml/2006/main}srgbClr', {'val': '5B6575'})
                ET.SubElement(rPr2, '{http://schemas.openxmlformats.org/drawingml/2006/main}latin', {'typeface': 'Calibri'})
                t2 = ET.SubElement(r2, '{http://schemas.openxmlformats.org/drawingml/2006/main}t')
                t2.text = "Graham Hall: 270-image exterior subset; 12 portrait images skipped; runtime 7,657.0 s (baseline) vs 6,860.7 s (Variant E, incremental extraction)."
                
                txBody.append(p1)
                txBody.append(p2)
                
                # Slightly expand height of note box to accommodate two lines comfortably
                xfrm = sp.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}xfrm')
                if xfrm is not None:
                    ext = xfrm.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ext')
                    if ext is not None:
                        ext.attrib['cy'] = '548640'  # 0.60 inches
                print("Slide 7: Updated footnote banner with verified Phase 5 vs Variant E distinction.")
                
    files['ppt/slides/slide7.xml'] = ET.tostring(s7_tree, encoding='utf-8', xml_declaration=True)
    
    # -------------------------------------------------------------
    # 4. Update Slide 8 (Table Values, Image Insertion, Interpretation Layout)
    # -------------------------------------------------------------
    s8_tree = ET.fromstring(files['ppt/slides/slide8.xml'])
    
    # Update table rows 4 and 5
    s8_tbl = s8_tree.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}tbl')
    if s8_tbl is not None:
        rows = s8_tbl.findall('{http://schemas.openxmlformats.org/drawingml/2006/main}tr')
        
        # Row 4: Verified inliers
        r4_tcs = rows[4].findall('{http://schemas.openxmlformats.org/drawingml/2006/main}tc')
        r4_c1_t = r4_tcs[1].find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}t')
        r4_c2_t = r4_tcs[2].find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}t')
        r4_c3_t = r4_tcs[3].find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}t')
        if r4_c1_t is not None: r4_c1_t.text = '1,343,083'
        if r4_c2_t is not None: r4_c2_t.text = '1,341,551'
        if r4_c3_t is not None: r4_c3_t.text = '\u22121,532 (\u22120.114%)'
        
        # Row 5: Mean track length
        r5_tcs = rows[5].findall('{http://schemas.openxmlformats.org/drawingml/2006/main}tc')
        r5_c1_t = r5_tcs[1].find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}t')
        r5_c2_t = r5_tcs[2].find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}t')
        r5_c3_t = r5_tcs[3].find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}t')
        if r5_c1_t is not None: r5_c1_t.text = '4.7216'
        if r5_c2_t is not None: r5_c2_t.text = '4.7149'
        if r5_c3_t is not None: r5_c3_t.text = '\u22120.0067 (\u22120.142%)'
        print("Slide 8: Verified inliers and mean track length populated in table.")
        
    # Adjust Interpretation shape (id=15 and id=16)
    for sp in s8_tree.findall('.//{http://schemas.openxmlformats.org/presentationml/2006/main}sp'):
        nv = sp.find('.//{http://schemas.openxmlformats.org/presentationml/2006/main}cNvPr')
        if nv is not None:
            sid = nv.attrib.get('id')
            if sid == '15':  # Interpretation background card
                xfrm = sp.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}xfrm')
                ext = xfrm.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ext')
                ext.attrib['cy'] = '1325880'  # 1.45 inches
            elif sid == '16':  # Interpretation text box
                xfrm = sp.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}xfrm')
                off = xfrm.find('{http://schemas.openxmlformats.org/drawingml/2006/main}off')
                ext = xfrm.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ext')
                off.attrib['y'] = '2816352'   # 3.08 inches
                ext.attrib['cy'] = '1188720'  # 1.30 inches
                
                # Adjust font sizes of interpretation text
                paragraphs = sp.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}p')
                if len(paragraphs) >= 2:
                    p1_rPr = paragraphs[0].find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}rPr')
                    if p1_rPr is not None:
                        p1_rPr.attrib['sz'] = '1300'
                    p2_rPr = paragraphs[1].find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}rPr')
                    if p2_rPr is not None:
                        p2_rPr.attrib['sz'] = '1000'
                print("Slide 8: Compacted Interpretation card to make room for visual comparison.")

    # Now add the Visual Comparison card and image
    spTree = s8_tree.find('.//{http://schemas.openxmlformats.org/presentationml/2006/main}spTree')
    
    # 4a. Container card for visual comparison
    card_xml = """
    <p:sp xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <p:nvSpPr>
        <p:cNvPr id="50" name="Visual Container" />
        <p:cNvSpPr />
        <p:nvPr />
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm>
          <a:off x="7589520" y="4206240" />
          <a:ext cx="4023360" cy="2103120" />
        </a:xfrm>
        <a:prstGeom prst="roundRect">
          <a:avLst>
            <a:gd name="adj" fmla="val 2778" />
          </a:avLst>
        </a:prstGeom>
        <a:solidFill>
          <a:srgbClr val="FFFFFF" />
        </a:solidFill>
        <a:ln w="12700">
          <a:solidFill>
            <a:srgbClr val="DDE5F0" />
          </a:solidFill>
          <a:prstDash val="solid" />
        </a:ln>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" />
        <a:lstStyle />
        <a:p><a:endParaRPr lang="en-US" sz="1000" /></a:p>
      </p:txBody>
    </p:sp>
    """
    spTree.append(ET.fromstring(card_xml))
    
    # 4b. Title label for visual comparison
    title_xml = """
    <p:sp xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <p:nvSpPr>
        <p:cNvPr id="51" name="Visual Title" />
        <p:cNvSpPr txBox="1" />
        <p:nvPr />
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm>
          <a:off x="7726680" y="4260000" />
          <a:ext cx="3749040" cy="260000" />
        </a:xfrm>
        <a:prstGeom prst="rect"><a:avLst /></a:prstGeom>
        <a:noFill /><a:ln />
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" />
        <a:lstStyle />
        <a:p>
          <a:pPr marL="0" indent="0" />
          <a:r>
            <a:rPr lang="en-US" sz="1050" b="1" dirty="0">
              <a:solidFill><a:srgbClr val="0F1D38" /></a:solidFill>
              <a:latin typeface="Calibri" />
            </a:rPr>
            <a:t>Visual Sample: Original (Left) vs. Variant E (Right)</a:t>
          </a:r>
        </a:p>
      </p:txBody>
    </p:sp>
    """
    spTree.append(ET.fromstring(title_xml))
    
    # 4c. Picture element
    pic_xml = """
    <p:pic xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <p:nvPicPr>
        <p:cNvPr id="52" name="Preview_IMG_0389" descr="Graham Hall Variant E Before/After Preview" />
        <p:cNvPicPr>
          <a:picLocks noChangeAspect="1" />
        </p:cNvPicPr>
        <p:nvPr />
      </p:nvPicPr>
      <p:blipFill>
        <a:blip r:embed="rId3" />
        <a:stretch>
          <a:fillRect />
        </a:stretch>
      </p:blipFill>
      <p:spPr>
        <a:xfrm>
          <a:off x="7726680" y="4550000" />
          <a:ext cx="3749040" cy="1249680" />
        </a:xfrm>
        <a:prstGeom prst="rect"><a:avLst /></a:prstGeom>
        <a:ln w="9525">
          <a:solidFill><a:srgbClr val="CBD5E1" /></a:solidFill>
          <a:prstDash val="solid" />
        </a:ln>
      </p:spPr>
    </p:pic>
    """
    spTree.append(ET.fromstring(pic_xml))
    
    # 4d. Caption below picture
    caption_xml = """
    <p:sp xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <p:nvSpPr>
        <p:cNvPr id="53" name="Visual Caption" />
        <p:cNvSpPr txBox="1" />
        <p:nvPr />
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm>
          <a:off x="7726680" y="5830000" />
          <a:ext cx="3749040" cy="400000" />
        </a:xfrm>
        <a:prstGeom prst="rect"><a:avLst /></a:prstGeom>
        <a:noFill /><a:ln />
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" />
        <a:lstStyle />
        <a:p>
          <a:pPr marL="0" indent="0" />
          <a:r>
            <a:rPr lang="en-US" sz="880" dirty="0">
              <a:solidFill><a:srgbClr val="4B5563" /></a:solidFill>
              <a:latin typeface="Calibri" />
            </a:rPr>
            <a:t>Graham Hall IMG_0389.jpg: Selective CLAHE (clip=1.2, grid=8x8). Contrast: 43.16 \u2192 47.42.</a:t>
          </a:r>
        </a:p>
      </p:txBody>
    </p:sp>
    """
    spTree.append(ET.fromstring(caption_xml))
    
    files['ppt/slides/slide8.xml'] = ET.tostring(s8_tree, encoding='utf-8', xml_declaration=True)
    print("Slide 8: Inserted visual comparison container, title, picture, and caption.")
    
    # -------------------------------------------------------------
    # 5. Update slide8.xml.rels & Add Media File
    # -------------------------------------------------------------
    s8_rels_tree = ET.fromstring(files['ppt/slides/_rels/slide8.xml.rels'])
    rel_elem = ET.Element('{http://schemas.openxmlformats.org/package/2006/relationships}Relationship', {
        'Id': 'rId3',
        'Type': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image',
        'Target': '../media/preview_IMG_0389.jpg'
    })
    s8_rels_tree.append(rel_elem)
    files['ppt/slides/_rels/slide8.xml.rels'] = ET.tostring(s8_rels_tree, encoding='utf-8', xml_declaration=True)
    
    with open(PREVIEW_IMG, 'rb') as f:
        files['ppt/media/preview_IMG_0389.jpg'] = f.read()
    print("Added preview_IMG_0389.jpg to ppt/media/ and linked in slide8.xml.rels.")
    
    # -------------------------------------------------------------
    # 6. Update Slide 9 Subtitle
    # -------------------------------------------------------------
    s9_tree = ET.fromstring(files['ppt/slides/slide9.xml'])
    for t_elem in s9_tree.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t'):
        if t_elem.text and 'Both axes show change relative to the baseline run of the same dataset.' in t_elem.text:
            t_elem.text = 'Left: % Point Change | Right: Reprojection Error Change'
            print("Slide 9: Updated subtitle to accurately describe the two separate bar charts.")
    files['ppt/slides/slide9.xml'] = ET.tostring(s9_tree, encoding='utf-8', xml_declaration=True)
    
    # -------------------------------------------------------------
    # 7. Update Slide 12 References URL
    # -------------------------------------------------------------
    s12_tree = ET.fromstring(files['ppt/slides/slide12.xml'])
    for t_elem in s12_tree.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t'):
        if t_elem.text and 'Datasets: [https://demuc.de/colmap/datasets/]' in t_elem.text:
            t_elem.text = 'Datasets: https://demuc.de/colmap/datasets/'
            print("Slide 12: Removed bracket placeholder from dataset URL.")
    files['ppt/slides/slide12.xml'] = ET.tostring(s12_tree, encoding='utf-8', xml_declaration=True)
    
    # -------------------------------------------------------------
    # Write Out Target PPTX
    # -------------------------------------------------------------
    with zipfile.ZipFile(TARGET_PPTX, 'w', compression=zipfile.ZIP_DEFLATED) as out_zip:
        for name, data in files.items():
            out_zip.writestr(name, data)
            
    print(f"Successfully generated revised presentation: {TARGET_PPTX}")
    print(f"Target file size: {os.path.getsize(TARGET_PPTX):,} bytes")

if __name__ == '__main__':
    update_presentation()

