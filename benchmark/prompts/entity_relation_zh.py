input_text = "{input_text}"
entity_types = "{entity_types}"

template = f"""---目标---
        从给定的文本文档中识别所有指定类型的实体，并识别实体间的关系。

        ---要求---
        请严格按照以下JSON格式输出结果，不要添加任何额外的文字说明：

        {{
        "entities": [
            {{
            "name": "实体名称",
            "type": "实体类型(person/organization/geo/event/category)",
            "description": "实体描述"
            }}
        ],
        "relationships": [
            {{
            "source": "源实体名称",
            "target": "目标实体名称", 
            "type": "关系类型(FAMILY/WORK/FRIEND/LOCATION/TEMPORAL/TOPIC/RELATED_TO)",
            "description": "关系描述",
            "strength": 0.8
            }}
        ],
        "keywords": ["关键词1", "关键词2"]
        }}

        ---示例---
        文本: "Caroline住在纽约，是心理咨询师。Melanie住在加州，是艺术家。她们是好朋友。"

        输出:
        {{
        "entities": [
            {{"name": "Caroline", "type": "person", "description": "住在纽约的心理咨询师"}},
            {{"name": "Melanie", "type": "person", "description": "住在加州的艺术家"}},
            {{"name": "纽约", "type": "geo", "description": "城市"}},
            {{"name": "加州", "type": "geo", "description": "州"}}
        ],
        "relationships": [
            {{"source": "Caroline", "target": "Melanie", "type": "FRIEND", "description": "好朋友关系", "strength": 0.9}},
            {{"source": "Caroline", "target": "纽约", "type": "LOCATION", "description": "居住地", "strength": 0.8}},
            {{"source": "Melanie", "target": "加州", "type": "LOCATION", "description": "居住地", "strength": 0.8}}
        ],
        "keywords": ["朋友关系", "职业", "地理位置"]
        }}

        ---实际任务---
        文本: {input_text}
        实体类型限制: {entity_types}

        请输出JSON格式的结果:"""