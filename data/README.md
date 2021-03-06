# 数据说明

## event_id.txt
* 格式：[new_event_id] [old_event_id]
* 说明：重新对events进行序号化

## member_id.txt
* 格式：[new_member_id] [old_member_id]
* 说明：重新对members进行序号化

## event2member.txt
* 格式：[new_event_id] [new_member_id1] [new_member_id2] [new_member_id3] ...
* 说明：记录每个event由哪些members参加

## member2topic.txt
* 格式：[new_member_id] [topic_id1] [topic_id2] [topic_id3] ...
* 说明：记录每个member参加了哪些topics

## topic.json
* 格式：
  ```json
  {
      "topic_id": {
          "urlkey": "xxx",
          "name": "xxx"
      }
  }
  ```
* 记录每个topic的相关信息

## group2member.txt
* 格式：[group_id] [new_member_id1] [new_member_id2] [new_member_id3] ...
* 说明：记录每个group包含哪些members（仅包含在RSVPs中出现的）

## event2group.txt
* 格式：[new_event_id] [group_id] 
* 说明：每个event对应的group

## event.json
* 格式：
  ```json
    {
        "event_id": {
          "group_id": "xxx", 
          "created_time": "xxx", 
          "desc": "xxx",
          "name": "xxx"
        }
    }
  ```
* 说明：event的相关信息