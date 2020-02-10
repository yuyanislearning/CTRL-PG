from tidylib import tidy_document
import xml.etree.ElementTree as ET
import tempfile
import os

class evaluation:

	def __init__(document_id, xml_folder, dev=True): #label_dict

		self.events = []
		self.timex3 = []
		self.document_id = document_id
		#self.label_dict = label_dict
		self.dev = dev
		self.xml_folder = xml_folder

	def eval(self, preds, event_ids):

		# read the corresponding xml file without tlinks
		xml_file = open(os.path.join(self.xml_folder, str(self.document_id)+'.xml'), 'r')
		document, _ = tidy_document(xml_file.read(), {"input_xml": True})

		with tempfile.NamedTemporaryFile() as temp:
			temp.write(document)
			temp.seek(0)
			tree = ET.parse(temp)
			root = tree.getroot()
			self.parseTags(root[1])

		# et = ET.parse(xml_file)
		# for i,(id1, id2, label_ids) in enumerate(zip(event_ids, preds)):
		# 	label = self.label_dict[label_ids]
		# 	event1 = self.events[id1] if "E" in id1 else self.timex3[id1]
		# 	event2 = self.events[id2] if "E" in id1 else self.timex3[id2]
		# 	new_tag = ET.SubElement(et.getroot(), 'TLINK')
		# 	new_tag.attrib['id'] = "TL"+str(i)
		# 	new_tag.attrib['fromID'] = id1
		# 	new_tag.attrib['fromText'] = event1
		# 	new_tag.attrib['toID = id2']
		# 	new_tag.attrib['toText'] = event2
		# 	new_tag.attrib['type'] = label
		# 	et.write(xml_file)
	
		xmlfile = open(os.path.join(self.xml_folder, str(self.document_id)+'.xml'), 'r')
		lines = xmlfile.readlines()
		writefile = open(os.path.join(self.xml_folder, str(self.document_id)+'.xml'), 'w')
		for line in lines:
			if "</TAGS>" not in line and "<TLINK" not in the line:
				writefile.write(line)
			else:
				for i,([id1, id2], label) in enumerate(zip(event_ids, preds)):

					#label = label_dict[label_ids]
					event1 = self.events[id1] if "E" in id1 else self.timex3[id1]
					event2 = self.events[id2] if "E" in id1 else self.timex3[id2]

					writefile.write('<TLINK id="TL{}" fromID="{}" fromText="{}" toID="{}" toText="{}" type="{}" />'.format(str(i), id1, event1, id2, event2, label) + '\n')
				writefile.write(line)
		writefile.close()
		

	def parseTags(self, tree):

		for child in tags:
			if child.tag == 'EVENT':
				self.events[child.attrib['id']] = child.attrib['text']  

			elif child.tag == 'TIMEX3':
				self.timex3[child.attrib['id']] = child.attrib['text'] 


