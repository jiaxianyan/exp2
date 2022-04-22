import xml.etree.ElementTree as ET
from glob import glob
import json
import os
from bs4 import BeautifulSoup

dataset_path = './Meetup/All_Unpack/'
output_path = './data/'


class Parser(object):
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.event2member_path = os.path.join(output_path, 'event2member.txt')
        self.group2member_path = os.path.join(output_path, 'group2member.txt')
        self.event_id_path = os.path.join(output_path, 'event_id.txt')
        self.member_id_path = os.path.join(output_path, 'member_id.txt')

    def file2dict(self, file_path, reverse=False):
        D = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                new_id, old_id = line.split()
                if reverse:
                    D[new_id] = old_id
                else:
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
            global_group_members = {}

        for xml_file in RSVPs_xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            event_ids = []
            member_ids = []
            group_id = ''
            for item in root[1]:
                for child in item:
                    if child.tag == 'member':
                        if mode == 'buildGraph':
                            member_ids.append(member_id_dict[child[0].text])
                        elif mode == 'getID':
                            member_ids.append(child[0].text)
                    elif child.tag == 'event':
                        if mode == 'buildGraph':
                            event_ids.append(event_id_dict[child[0].text])
                        elif mode == 'getID':
                            event_ids.append(child[0].text)
                    elif child.tag == 'group':
                        if mode == 'buildGraph':
                            group_id = child[2].text
            if (len(set(event_ids)) == 1):
                if mode == 'buildGraph':
                    with open(self.event2member_path, 'a', encoding='utf-8') as f:
                        f.write(
                            f'{event_ids[0]} {" ".join(sorted(set(member_ids)))}\n')
                    if not group_id in global_group_members:
                        global_group_members[group_id] = member_ids
                    else:
                        global_group_members[group_id].extend(member_ids)
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
            with open(self.group2member_path, 'w', encoding='utf-8') as f:
                for group_id, members in global_group_members.items():
                    f.write(f'{group_id} {" ".join(members)}\n')

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

    def getEventInfo(self, event_file, event2group_file):
        event_path = os.path.join(self.output_path, event_file)
        event2group_path = os.path.join(self.output_path, event2group_file)
        global_events = {}
        event2group = {}
        xml_files = glob(os.path.join(
            self.dataset_path, 'PastEvent *.xml'))
        event_id_dict = self.file2dict(self.event_id_path)
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            event_id = ''
            desc = ''
            name = ''
            group_id = ''
            created_time = 0
            utc_offset = 0

            for child in root:
                if child.tag == 'group':
                    group_id = child[3].text
                elif child.tag == 'id':
                    event_id = event_id_dict[child.text]
                elif child.tag == 'created':
                    created_time = int(child.text)
                elif child.tag == 'utc_offset':
                    utc_offset = int(child.text)
                elif child.tag == 'description':
                    desc = child.text
                    desc = BeautifulSoup(desc, "html.parser").text
                elif child.tag == 'name':
                    name = child.text
            global_events[event_id] = {
                'created_time': created_time+utc_offset, 'desc': desc, 'name': name}
            event2group[event_id] = group_id

        with open(event_path, 'w', encoding='utf-8') as f:
            json.dump(global_events, f)

        with open(event2group_path, 'w', encoding='utf-8') as f:
            for event_id, group_id in event2group.items():
                f.write(f'{event_id} {group_id}\n')


if __name__ == '__main__':
    parser = Parser(dataset_path, output_path)
    parser.parseID('member_id.txt', 'Memeber')
    parser.parseID('event_id.txt', 'PastEvent')
    parser.parseEvent2Member(mode='getID')
    members_num = parser.addNewID('member_id.txt')
    parser.addNewID('event_id.txt', members_num)
    parser.parseEvent2Member()
    parser.getTopics('topic.json', 'member2topic.txt')
    parser.getEventInfo('event.json', 'event2group.txt')
