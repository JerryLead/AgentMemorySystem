import os
import json
import argparse
import time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    print("[WARN] 未检测到 tqdm 库，建议通过 pip install tqdm 安装以获得进度条体验！")


def tqdm_wrap(iterable, **kwargs):
    if tqdm is not None:
        return tqdm(iterable, **kwargs)
    else:
        return iterable


from core.MemoryUnit import MemoryUnit
from core.MemorySpace import MemorySpace
from core.SemanticMap import SemanticMap
from core.SemanticGraph import SemanticGraph

yelp_base_path = "/mnt/data1/home/guozy/gzy/datasets/yelp"
yelp_dataset_path = os.path.join(yelp_base_path, "yelp_dataset")
yelp_photos_path = os.path.join(yelp_base_path, "yelp_photos/photos")
data_save_dir = os.path.join("data", "yelp")

business_json = os.path.join(
    yelp_dataset_path, "yelp_academic_dataset_business.json"
)  # 商家
checkin_json = os.path.join(yelp_dataset_path, "yelp_academic_dataset_checkin.json")
review_json = os.path.join(
    yelp_dataset_path, "yelp_academic_dataset_review.json"
)  # 评论
tip_json = os.path.join(yelp_dataset_path, "yelp_academic_dataset_tip.json")  # 提示
user_json = os.path.join(yelp_dataset_path, "yelp_academic_dataset_user.json")  # 用户
photo_json = os.path.join(yelp_base_path, "yelp_photos/photos.json")  # 照片

YELP_IMPORT_LIMIT = 1666  # None 表示全量导入，否则为采样上限


def count_lines(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def load_json_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[ERROR] 解析第{idx+1}行失败: {e}")
                continue


def import_yelp_data():
    start_time = time.time()
    print("[INFO] 开始导入Yelp数据...")
    smap = SemanticMap(
        image_embedding_model_name="/mnt/data1/home/guozy/gzy/models/clip-ViT-B-32",
        text_embedding_model_name="/mnt/data1/home/guozy/gzy/models/clip-ViT-B-32-multilingual-v1",
    )
    sgraph = SemanticGraph(smap)
    connected = sgraph.connect_to_neo4j(database="neo4j", milvus_collection="yelp")
    print(f"Connected to DB: {connected}")

    # 创建MemorySpace
    business_space = MemorySpace("business")
    review_space = MemorySpace("review")
    user_space = MemorySpace("user")
    photo_space = MemorySpace("photo")
    checkin_space = MemorySpace("checkin")
    tip_space = MemorySpace("tip")

    # 注册MemorySpace
    sgraph.add_memory_space_in_map(business_space)
    sgraph.add_memory_space_in_map(review_space)
    sgraph.add_memory_space_in_map(user_space)
    sgraph.add_memory_space_in_map(photo_space)
    sgraph.add_memory_space_in_map(checkin_space)
    sgraph.add_memory_space_in_map(tip_space)

    # 处理business数据
    business_total = count_lines(business_json)
    business_count = 0
    for i, item in enumerate(
        tqdm_wrap(
            load_json_lines(business_json), total=business_total, desc="导入business"
        )
    ):
        if YELP_IMPORT_LIMIT is not None and i >= YELP_IMPORT_LIMIT:
            break
        try:
            mu = MemoryUnit(
                uid=f"business_{item.get('business_id')}",
                raw_data={**item},
                metadata={"type": "business"},
            )
            sgraph.add_unit(mu, space_names=["business"])
            business_count += 1
        except Exception as e:
            print(f"[ERROR] business第{business_count+1}条处理异常: {e}")
            raise
    print(f"[INFO] business导入数量: {business_count}")

    # 处理review数据
    review_total = count_lines(review_json)
    review_count = 0
    for i, item in enumerate(
        tqdm_wrap(load_json_lines(review_json), total=review_total, desc="导入review")
    ):
        if YELP_IMPORT_LIMIT is not None and i >= YELP_IMPORT_LIMIT:
            break
        try:
            mu = MemoryUnit(
                uid=f"review_{item.get('review_id')}",
                raw_data={
                    "text_content": item.get("text"),
                    **item,
                },
                metadata={
                    "type": "review",
                    "user_id": item.get("user_id"),
                    "business_id": item.get("business_id"),
                },
            )
            sgraph.add_unit(mu, space_names=["review"])
            review_count += 1
        except Exception as e:
            print(f"[ERROR] review第{review_count+1}条处理异常: {e}")
            raise
    print(f"[INFO] review导入数量: {review_count}")

    # 处理user数据
    user_total = count_lines(user_json)
    user_count = 0
    for i, item in enumerate(
        tqdm_wrap(load_json_lines(user_json), total=user_total, desc="导入user")
    ):
        if YELP_IMPORT_LIMIT is not None and i >= YELP_IMPORT_LIMIT:
            break
        try:
            mu = MemoryUnit(
                uid=f"user_{item.get('user_id')}",
                raw_data={**item},
                metadata={
                    "type": "user",
                    "friends_id": item.get("friends"),
                },
            )
            sgraph.add_unit(mu, space_names=["user"])
            user_count += 1
        except Exception as e:
            print(f"[ERROR] user第{user_count+1}条处理异常: {e}")
            raise
    print(f"[INFO] user导入数量: {user_count}")

    # 处理checkin数据
    checkin_total = count_lines(checkin_json)
    checkin_count = 0
    for i, item in enumerate(
        tqdm_wrap(
            load_json_lines(checkin_json), total=checkin_total, desc="导入checkin"
        )
    ):
        if YELP_IMPORT_LIMIT is not None and i >= YELP_IMPORT_LIMIT:
            break
        try:
            mu = MemoryUnit(
                uid=f"checkin_{item.get('business_id')}",
                raw_data={**item},
                metadata={
                    "type": "checkin",
                    "business_id": item.get("business_id"),
                },
            )
            sgraph.add_unit(mu, space_names=["checkin"])
            checkin_count += 1
        except Exception as e:
            print(f"[ERROR] checkin第{checkin_count+1}条处理异常: {e}")
            raise
    print(f"[INFO] checkin导入数量: {checkin_count}")

    # 处理tip数据
    tip_total = count_lines(tip_json)
    tip_count = 0
    for i, item in enumerate(
        tqdm_wrap(load_json_lines(tip_json), total=tip_total, desc="导入tip")
    ):
        if YELP_IMPORT_LIMIT is not None and i >= YELP_IMPORT_LIMIT:
            break
        try:
            mu = MemoryUnit(
                uid=f"tip_{item.get('user_id')}_{item.get('business_id')}",
                raw_data={
                    "text_content": item.get("text"),
                    **item,
                },
                metadata={
                    "type": "tip",
                    "user_id": item.get("user_id"),
                    "business_id": item.get("business_id"),
                },
            )
            sgraph.add_unit(mu, space_names=["tip"])
            tip_count += 1
        except Exception as e:
            print(f"[ERROR] tip第{tip_count+1}条处理异常: {e}")
            raise
    print(f"[INFO] tip导入数量: {tip_count}")

    # 处理photo数据
    photo_total = count_lines(photo_json)
    photo_count = 0
    for i, item in enumerate(
        tqdm_wrap(load_json_lines(photo_json), total=photo_total, desc="导入photo")
    ):
        if YELP_IMPORT_LIMIT is not None and i >= YELP_IMPORT_LIMIT:
            break
        try:
            mu = MemoryUnit(
                uid=f"photo_{item.get('photo_id')}",
                raw_data={
                    "image_path": os.path.join(
                        yelp_photos_path, f"{item.get('photo_id')}.jpg"
                    ),
                    **item,
                },
                metadata={
                    "type": "photo",
                    "business_id": item.get("business_id"),
                    "label": item.get("label"),
                },
            )
            sgraph.add_unit(mu, space_names=["photo"])
            photo_count += 1
        except Exception as e:
            print(f"[ERROR] photo第{photo_count+1}条处理异常: {e}")
            raise
    print(f"[INFO] photo导入数量: {photo_count}")

    # 建立关系
    print("[INFO] 正在建立关系...")
    friend_rel_count = 0
    for mu in user_space.get_all_units():
        user_id = mu.raw_data.get("user_id")
        friends = mu.raw_data.get("friends")
        if friends and isinstance(friends, str):
            for friend_id in friends.split(", "):
                if friend_id:
                    sgraph.add_relationship(
                        f"user_{user_id}",
                        f"user_{friend_id}",
                        "friend",
                        bidirectional=True,
                    )
                    friend_rel_count += 1
    print(f"[INFO] friend关系建立数量: {friend_rel_count}")

    tip_rel_count = 0
    tip_space = smap.get_memory_space("tip")
    if tip_space:
        for mu in tip_space.get_all_units():
            user_id = mu.metadata.get("user_id")
            business_id = mu.metadata.get("business_id")
            tip_uid = mu.uid
            if user_id:
                sgraph.add_relationship(f"user_{user_id}", tip_uid, "write_tip")
                tip_rel_count += 1
            if business_id:
                sgraph.add_relationship(tip_uid, f"business_{business_id}", "tip_for")
                tip_rel_count += 1
    print(f"[INFO] tip相关关系建立数量: {tip_rel_count}")

    review_rel_count = 0
    review_space = smap.get_memory_space("review")
    if review_space:
        for mu in review_space.get_all_units():
            user_id = mu.metadata.get("user_id")
            business_id = mu.metadata.get("business_id")
            review_uid = mu.uid
            if user_id:
                sgraph.add_relationship(f"user_{user_id}", review_uid, "write_review")
                review_rel_count += 1
            if business_id:
                sgraph.add_relationship(
                    review_uid, f"business_{business_id}", "review_for"
                )
                review_rel_count += 1
    print(f"[INFO] review相关关系建立数量: {review_rel_count}")

    photo_rel_count = 0
    photo_space = smap.get_memory_space("photo")
    if photo_space:
        for mu in photo_space.get_all_units():
            business_id = mu.metadata.get("business_id")
            photo_uid = mu.uid
            if business_id:
                sgraph.add_relationship(
                    photo_uid, f"business_{business_id}", "photo_of"
                )
                photo_rel_count += 1
    print(f"[INFO] photo相关关系建立数量: {photo_rel_count}")

    checkin_rel_count = 0
    checkin_space = smap.get_memory_space("checkin")
    if checkin_space:
        for mu in checkin_space.get_all_units():
            business_id = mu.metadata.get("business_id")
            checkin_uid = mu.uid
            if business_id:
                sgraph.add_relationship(
                    checkin_uid, f"business_{business_id}", "checkin_at"
                )
                checkin_rel_count += 1
    print(f"[INFO] checkin相关关系建立数量: {checkin_rel_count}")

    print("[INFO] 正在建立索引...")
    sgraph.build_semantic_map_index()

    # 保存
    print("[INFO] 正在保存图谱...")
    os.makedirs(data_save_dir, exist_ok=True)
    sgraph.save_graph(data_save_dir)
    print(f"[INFO] Yelp数据导入并保存到 {data_save_dir} 完成！")
    print(f"[INFO] 总用时: {time.time() - start_time:.2f} 秒")


def load_yelp_graph():
    """加载已保存的SemanticGraph并打印摘要"""
    if not os.path.exists(data_save_dir):
        print(f"[ERROR] 数据目录 {data_save_dir} 不存在，请先导入数据！")
        return
    print(f"[INFO] 正在加载 {data_save_dir} ...")
    semantic_graph = SemanticGraph.load_graph(
        data_save_dir,
        image_embedding_model_name="/mnt/data1/home/guozy/gzy/models/clip-ViT-B-32",
        text_embedding_model_name="/mnt/data1/home/guozy/gzy/models/clip-ViT-B-32-multilingual-v1",
    )
    print("[INFO] 加载完成，图谱摘要如下：")
    semantic_graph.display_graph_summary()

    # 简单向量检索测试
    print("[TEST] 开始简单向量检索测试（文本相似性）...")
    query_text = "good food"
    results = semantic_graph.search_similarity_in_graph(
        query_text=query_text, top_k=3, ms_names=["review"]
    )
    for i, (unit, score) in enumerate(results):
        print(
            f"[TEST] Top {i+1} review: uid={unit.uid}, score={score:.4f}, text={unit.raw_data.get('text_content', '')}"
        )

    # 简单标量检索测试
    print("[TEST] 开始简单标量检索测试（business stars >= 4.0）...")
    filter_cond = {"stars": {"gte": 4.0}}
    business_units = semantic_graph.filter_memory_units(
        ms_names=["business"], filter_condition=filter_cond
    )
    for i, unit in enumerate(business_units[:3]):
        print(
            f"[TEST] Top {i+1} business: uid={unit.uid}, stars={unit.raw_data.get('stars')}, name={unit.raw_data.get('name')}"
        )


def main():
    parser = argparse.ArgumentParser(description="Yelp数据导入与加载工具")
    parser.add_argument(
        "-i", "--import_", action="store_true", help="导入Yelp原始数据并保存图谱"
    )
    parser.add_argument(
        "-l", "--load", action="store_true", help="加载已保存的Yelp图谱并打印摘要"
    )
    args = parser.parse_args()

    if args.import_:
        import_yelp_data()
    else:
        load_yelp_graph()


if __name__ == "__main__":
    main()

# 导入数据
# python -m example.yelp.yelp -i
# 加载数据
# python -m example.yelp.yelp
