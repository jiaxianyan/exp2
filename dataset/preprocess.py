import xml.etree.ElementTree as ET
from glob import glob
import json
import os

dataset_path = './Meetup/All_Unpack/'
output_path = './data/'


class Parser(object):
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.event2member_path = os.path.join(output_path, 'event2member.txt')
        self.event_id_path = os.path.join(output_path, 'event_id.txt')
        self.member_id_path = os.path.join(output_path, 'member_id.txt')

    def file2dict(self, file_path):
        D = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                new_id, old_id = line.split()
                D[old_id] = new_id
        return D

    def parseEvent2Member(self, mode='buildGraph'):
        # mode 'getID' means get new IDs from RSVPs
        # mode 'buildGraph' mean build relation between Member and Event

        if mode == 'buildGraph':
            event_id_dict = self.file2dict(self.event_id_path)
            member_id_dict = self.file2dict(self.member_id_path)

            if os.path.exists(self.event2member_path):
                os.remove(self.event2member_path)

        RSVPs_xml_files = glob(os.path.join(self.dataset_path, 'RSVPs *.xml'))
        if mode == 'getID':
            global_event_ids = []
            global_member_ids = []
        elif mode == 'buildGraph':
            f = open(self.event2member_path, 'a', encoding='utf-8')

        for xml_file in RSVPs_xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            event_ids = []
            member_ids = []
            for item in root[1]:
                for child in item:
                    if child.tag == 'member':
                        if mode == 'buildGraph':
                            member_ids.append(member_id_dict[child[0].text])
                        elif mode == 'getID':
                            member_ids.append(child[0].text)
                    if child.tag == 'event':
                        if mode == 'buildGraph':
                            event_ids.append(event_id_dict[child[0].text])
                        elif mode == 'getID':
                            event_ids.append(child[0].text)
            if (len(set(event_ids)) == 1):
                if mode == 'buildGraph':
                    with open(self.event2member_path, 'a', encoding='utf-8') as f:
                        f.write(
                            f'{event_ids[0]} {" ".join(sorted(set(member_ids)))}\n')
                elif mode == 'getID':
                    global_event_ids.append(event_ids[0])
                    global_member_ids.extend(member_ids)
            else:
                continue

        if mode == 'getID':
            global_event_ids = sorted(set(global_event_ids))
            global_member_ids = sorted(set(global_member_ids))

            with open(self.event_id_path, 'a', encoding='utf-8') as fe:
                fe.write('\n'.join(global_event_ids))
            with open(self.member_id_path, 'a', encoding='utf-8') as fm:
                fm.write('\n'.join(global_member_ids))
        elif mode == 'buildGraph':
            f.close()

    def parseID(self, file_path, file_pattern):
        id_path = os.path.join(self.output_path, file_path)
        xml_files = glob(os.path.join(
            self.dataset_path, f'{file_pattern} *.xml'))
        global_ids = []
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for child in root:
                if child.tag == 'id':
                    global_ids.append(child.text)
        global_ids = sorted(set(global_ids))
        with open(id_path, 'w', encoding='utf-8') as f:
            for id in global_ids:
                f.write(f'{id}\n')

    def addNewID(self, file_path, start=0):
        global_ids = []
        id_path = os.path.join(self.output_path, file_path)
        with open(id_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                global_ids.append(line)
        global_ids = sorted(set(global_ids))
        with open(id_path, 'w', encoding='utf-8') as f:
            for new_id, old_id in enumerate(global_ids, start):
                f.write(f'{new_id} {old_id}\n')
        return len(global_ids)

    def getTime(self, file_path, file_pattern):
        id_path = os.path.join(self.output_path, file_path)
        with open(id_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(id_path, 'w', encoding='utf-8') as f:
            for line in lines:
                new_id, old_id = line.split()
                old_id = old_id.strip()
                xml_file = os.path.join(
                    self.dataset_path, f'{file_pattern} {old_id}.xml')
                tree = ET.parse(xml_file)
                root = tree.getroot()
                created_time = 0
                utc_offset = 0
                for child in root:
                    if child.tag == 'created':
                        created_time = int(child.text)
                    elif child.tag == 'utc_offset':
                        utc_offset = int(child.text)
                f.write(f'{new_id} {old_id} {str(created_time + utc_offset)}\n')

    def getTopics(self, topics_file, member2topic_file):
        global_topics = {}
        topics_path = os.path.join(self.output_path, topics_file)
        member2topic_path = os.path.join(self.output_path, member2topic_file)

        if os.path.exists(member2topic_path):
            os.remove(member2topic_path)

        xml_files = glob(os.path.join(
            self.dataset_path, 'Memeber *.xml'))
        member_id_dict = self.file2dict(self.member_id_path)
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            member_id = ''
            member2_topics = []
            for child in root:
                if child.tag == 'topics':
                    for topic in child:
                        topic_id = topic[0].text
                        topic_urlkey = topic[1].text
                        topic_name = topic[2].text
                        global_topics[topic_id] = {
                            'urlkey': topic_urlkey, 'name': topic_name}
                        member2_topics.append(topic_id)
                elif child.tag == 'id':
                    member_id = member_id_dict[child.text]
            with open(member2topic_path, 'a', encoding='utf-8') as f:
                f.write(f'{member_id} {" ".join(member2_topics)}\n')

        print(f'There are {len(global_topics)} Topics! Saving...\n')
        with open(topics_path, 'w', encoding='utf-8') as f:
            json.dump(global_topics, f)


if __name__ == '__main__':
    parser = Parser(dataset_path, output_path)
    parser.parseID('member_id.txt', 'Memeber')
    parser.parseID('event_id.txt', 'PastEvent')
    parser.parseEvent2Member(mode='getID')
    members_num = parser.addNewID('member_id.txt')
    parser.addNewID('event_id.txt', members_num)
    parser.parseEvent2Member()
    parser.getTime('event_id.txt', 'PastEvent')
    parser.getTopics('topic.json', 'member2topic.txt')
