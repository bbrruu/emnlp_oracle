# Tokenizer 驗證報告（RQ1）

## Qwen/Qwen2.5-7B-Instruct

### 孤立形式切分

| 形式 | n_subtokens | pieces |
|---|---|---|
| '台灣' | 1 | 'åı°çģ£' |
| '臺灣' | 1 | 'èĩºçģ£' |
| '台湾' | 1 | 'åı°æ¹¾' |
| 'Taiwan' | 2 | 'Tai' | 'wan' |
| ' Taiwan' | 1 | 'ĠTaiwan' |
| 'Japan' | 1 | 'Japan' |
| ' Japan' | 1 | 'ĠJapan' |
| '日本' | 1 | 'æĹ¥æľ¬' |
| '冰島' | 2 | 'åĨ°' | 'å³¶' |
| 'Iceland' | 2 | 'I' | 'celand' |
| ' Iceland' | 1 | 'ĠIceland' |

### 語境內診斷摘要

- 句數：360；帶警告：0
- mention subtoken 數（lang, n）：[('en', 1), ('en', 2), ('en', 3), ('en', 4), ('zh', 1), ('zh', 2), ('zh', 3)]
- 樸素 sublist 搜尋與 offset 法不一致：186 句（不一致即為樸素法之失敗案例，抽取管線務必採 offset 法）
- 前導 < 8 tokens：181 句 → SUP-085/zh(3), SUP-085/en(3), SUP-086/zh(1), SUP-086/en(6), SUP-087/zh(7), SUP-087/en(7), SUP-088/zh(4), SUP-088/en(3), SUP-089/zh(3), SUP-089/en(3), SUP-090/zh(1), SUP-090/en(6), SUP-091/zh(7), SUP-091/en(7), SUP-092/zh(4), SUP-092/en(3), SUP-093/zh(3), SUP-093/en(3), SUP-094/zh(1), SUP-094/en(6), SUP-095/zh(6), SUP-095/en(7), SUP-096/zh(4), SUP-096/en(3), SUP-097/zh(3), SUP-097/en(3), SUP-098/zh(1), SUP-098/en(6), SUP-099/zh(7), SUP-099/en(7), SUP-100/zh(4), SUP-100/en(3), SUP-101/zh(3), SUP-101/en(3), SUP-102/zh(1), SUP-102/en(6), SUP-103/zh(7), SUP-103/en(7), SUP-104/zh(4), SUP-104/en(3), SUP-105/zh(3), SUP-105/en(3), SUP-106/zh(1), SUP-106/en(6), SUP-107/zh(7), SUP-107/en(7), SUP-108/zh(4), SUP-108/en(3), SUP-109/zh(4), SUP-109/en(3), SUP-110/zh(1), SUP-110/en(6), SUP-111/en(7), SUP-112/zh(5), SUP-112/en(3), SUP-113/zh(4), SUP-113/en(3), SUP-114/zh(1), SUP-114/en(6), SUP-115/en(7), SUP-116/zh(5), SUP-116/en(3), SUP-117/zh(4), SUP-117/en(3), SUP-118/zh(1), SUP-118/en(6), SUP-119/zh(7), SUP-119/en(7), SUP-120/zh(5), SUP-120/en(3), SUP-121/zh(4), SUP-121/en(3), SUP-122/zh(1), SUP-122/en(6), SUP-123/en(7), SUP-124/zh(5), SUP-124/en(3), SUP-125/zh(4), SUP-125/en(3), SUP-126/zh(1), SUP-126/en(6), SUP-127/en(7), SUP-128/zh(5), SUP-128/en(3), SUP-129/zh(4), SUP-129/en(3), SUP-130/zh(1), SUP-130/en(6), SUP-131/en(7), SUP-132/zh(5), SUP-132/en(3), SUP-133/zh(3), SUP-133/en(5), SUP-134/zh(1), SUP-135/zh(7), SUP-135/en(7), SUP-136/zh(4), SUP-136/en(5), SUP-137/zh(3), SUP-137/en(5), SUP-138/zh(1), SUP-139/zh(7), SUP-139/en(7), SUP-140/zh(4), SUP-140/en(5), SUP-141/zh(3), SUP-141/en(5), SUP-142/zh(1), SUP-143/zh(6), SUP-143/en(7), SUP-144/zh(4), SUP-144/en(5), SUP-145/zh(3), SUP-145/en(5), SUP-146/zh(1), SUP-147/zh(7), SUP-147/en(7), SUP-148/zh(4), SUP-148/en(5), SUP-149/zh(3), SUP-149/en(5), SUP-150/zh(1), SUP-151/zh(7), SUP-151/en(7), SUP-152/zh(4), SUP-152/en(5), SUP-153/zh(3), SUP-153/en(5), SUP-154/zh(1), SUP-155/zh(7), SUP-155/en(7), SUP-156/zh(4), SUP-156/en(5), SUP-157/zh(3), SUP-157/en(3), SUP-158/zh(1), SUP-158/en(6), SUP-159/zh(7), SUP-159/en(7), SUP-160/zh(4), SUP-160/en(3), SUP-161/zh(3), SUP-161/en(3), SUP-162/zh(1), SUP-162/en(6), SUP-163/zh(7), SUP-163/en(7), SUP-164/zh(4), SUP-164/en(3), SUP-165/zh(3), SUP-165/en(3), SUP-166/zh(1), SUP-166/en(6), SUP-167/zh(6), SUP-167/en(7), SUP-168/zh(4), SUP-168/en(3), SUP-169/zh(3), SUP-169/en(3), SUP-170/zh(1), SUP-170/en(6), SUP-171/zh(7), SUP-171/en(7), SUP-172/zh(4), SUP-172/en(3), SUP-173/zh(3), SUP-173/en(3), SUP-174/zh(1), SUP-174/en(6), SUP-175/zh(7), SUP-175/en(7), SUP-176/zh(4), SUP-176/en(3), SUP-177/zh(3), SUP-177/en(3), SUP-178/zh(1), SUP-178/en(6), SUP-179/zh(7), SUP-179/en(7), SUP-180/zh(4), SUP-180/en(3)
- 句對長度匹配（±20%）：149/180 通過 → 超標：SUP-086(gap=0.273), SUP-090(gap=0.273), SUP-094(gap=0.25), SUP-098(gap=0.231), SUP-103(gap=0.286), SUP-104(gap=0.25), SUP-106(gap=0.231), SUP-110(gap=0.333), SUP-112(gap=0.286), SUP-114(gap=0.333), SUP-116(gap=0.286), SUP-118(gap=0.308), SUP-120(gap=0.25), SUP-122(gap=0.286), SUP-124(gap=0.222), SUP-127(gap=0.214), SUP-130(gap=0.286), SUP-132(gap=0.222), SUP-135(gap=0.231), SUP-139(gap=0.231), SUP-143(gap=0.286), SUP-149(gap=0.267), SUP-151(gap=0.375), SUP-152(gap=0.4), SUP-158(gap=0.273), SUP-162(gap=0.273), SUP-166(gap=0.25), SUP-170(gap=0.231), SUP-175(gap=0.286), SUP-176(gap=0.25), SUP-178(gap=0.231)

## google/gemma-3-12b-it

### 孤立形式切分

| 形式 | n_subtokens | pieces |
|---|---|---|
| '台灣' | 1 | '台灣' |
| '臺灣' | 1 | '臺灣' |
| '台湾' | 1 | '台湾' |
| 'Taiwan' | 1 | 'Taiwan' |
| ' Taiwan' | 1 | '▁Taiwan' |
| 'Japan' | 1 | 'Japan' |
| ' Japan' | 1 | '▁Japan' |
| '日本' | 1 | '日本' |
| '冰島' | 2 | '冰' | '島' |
| 'Iceland' | 1 | 'Iceland' |
| ' Iceland' | 1 | '▁Iceland' |

### 語境內診斷摘要

- 句數：360；帶警告：0
- mention subtoken 數（lang, n）：[('en', 1), ('en', 2), ('en', 3), ('en', 4), ('zh', 1), ('zh', 2), ('zh', 3)]
- 樸素 sublist 搜尋與 offset 法不一致：184 句（不一致即為樸素法之失敗案例，抽取管線務必採 offset 法）
- 前導 < 8 tokens：186 句 → SUP-085/zh(3), SUP-085/en(3), SUP-086/zh(1), SUP-086/en(6), SUP-087/zh(5), SUP-087/en(7), SUP-088/zh(4), SUP-088/en(3), SUP-089/zh(3), SUP-089/en(3), SUP-090/zh(1), SUP-090/en(6), SUP-091/zh(5), SUP-091/en(7), SUP-092/zh(4), SUP-092/en(3), SUP-093/zh(3), SUP-093/en(3), SUP-094/zh(1), SUP-094/en(6), SUP-095/zh(5), SUP-095/en(7), SUP-096/zh(4), SUP-096/en(3), SUP-097/zh(3), SUP-097/en(3), SUP-098/zh(1), SUP-098/en(6), SUP-099/zh(5), SUP-099/en(7), SUP-100/zh(4), SUP-100/en(3), SUP-101/zh(3), SUP-101/en(3), SUP-102/zh(1), SUP-102/en(6), SUP-103/zh(5), SUP-103/en(7), SUP-104/zh(4), SUP-104/en(3), SUP-105/zh(3), SUP-105/en(3), SUP-106/zh(1), SUP-106/en(6), SUP-107/zh(5), SUP-107/en(7), SUP-108/zh(4), SUP-108/en(3), SUP-109/zh(3), SUP-109/en(3), SUP-110/zh(1), SUP-110/en(6), SUP-111/zh(5), SUP-111/en(7), SUP-112/zh(4), SUP-112/en(3), SUP-113/zh(3), SUP-113/en(3), SUP-114/zh(1), SUP-114/en(6), SUP-115/zh(5), SUP-115/en(7), SUP-116/zh(4), SUP-116/en(3), SUP-117/zh(3), SUP-117/en(3), SUP-118/zh(1), SUP-118/en(6), SUP-119/zh(5), SUP-119/en(7), SUP-120/zh(4), SUP-120/en(3), SUP-121/zh(3), SUP-121/en(3), SUP-122/zh(1), SUP-122/en(6), SUP-123/zh(5), SUP-123/en(7), SUP-124/zh(4), SUP-124/en(3), SUP-125/zh(3), SUP-125/en(3), SUP-126/zh(1), SUP-126/en(6), SUP-127/zh(5), SUP-127/en(7), SUP-128/zh(4), SUP-128/en(3), SUP-129/zh(3), SUP-129/en(3), SUP-130/zh(1), SUP-130/en(6), SUP-131/zh(5), SUP-131/en(7), SUP-132/zh(4), SUP-132/en(3), SUP-133/zh(3), SUP-133/en(5), SUP-134/zh(1), SUP-135/zh(5), SUP-135/en(7), SUP-136/zh(4), SUP-136/en(5), SUP-137/zh(3), SUP-137/en(5), SUP-138/zh(1), SUP-139/zh(5), SUP-139/en(7), SUP-140/zh(4), SUP-140/en(5), SUP-141/zh(3), SUP-141/en(5), SUP-142/zh(1), SUP-143/zh(5), SUP-143/en(7), SUP-144/zh(4), SUP-144/en(5), SUP-145/zh(3), SUP-145/en(5), SUP-146/zh(1), SUP-147/zh(5), SUP-147/en(7), SUP-148/zh(4), SUP-148/en(5), SUP-149/zh(3), SUP-149/en(5), SUP-150/zh(1), SUP-151/zh(5), SUP-151/en(7), SUP-152/zh(4), SUP-152/en(5), SUP-153/zh(3), SUP-153/en(5), SUP-154/zh(1), SUP-155/zh(5), SUP-155/en(7), SUP-156/zh(4), SUP-156/en(5), SUP-157/zh(3), SUP-157/en(3), SUP-158/zh(1), SUP-158/en(6), SUP-159/zh(5), SUP-159/en(7), SUP-160/zh(4), SUP-160/en(3), SUP-161/zh(3), SUP-161/en(3), SUP-162/zh(1), SUP-162/en(6), SUP-163/zh(5), SUP-163/en(7), SUP-164/zh(4), SUP-164/en(3), SUP-165/zh(3), SUP-165/en(3), SUP-166/zh(1), SUP-166/en(6), SUP-167/zh(5), SUP-167/en(7), SUP-168/zh(4), SUP-168/en(3), SUP-169/zh(3), SUP-169/en(3), SUP-170/zh(1), SUP-170/en(6), SUP-171/zh(5), SUP-171/en(7), SUP-172/zh(4), SUP-172/en(3), SUP-173/zh(3), SUP-173/en(3), SUP-174/zh(1), SUP-174/en(6), SUP-175/zh(5), SUP-175/en(7), SUP-176/zh(4), SUP-176/en(3), SUP-177/zh(3), SUP-177/en(3), SUP-178/zh(1), SUP-178/en(6), SUP-179/zh(5), SUP-179/en(7), SUP-180/zh(4), SUP-180/en(3)
- 句對長度匹配（±20%）：155/180 通過 → 超標：SUP-087(gap=0.273), SUP-091(gap=0.273), SUP-095(gap=0.25), SUP-099(gap=0.231), SUP-103(gap=0.357), SUP-107(gap=0.231), SUP-111(gap=0.273), SUP-115(gap=0.273), SUP-119(gap=0.25), SUP-123(gap=0.231), SUP-127(gap=0.357), SUP-131(gap=0.231), SUP-135(gap=0.385), SUP-139(gap=0.385), SUP-143(gap=0.357), SUP-147(gap=0.333), SUP-151(gap=0.438), SUP-152(gap=0.3), SUP-155(gap=0.333), SUP-159(gap=0.273), SUP-163(gap=0.273), SUP-167(gap=0.25), SUP-171(gap=0.231), SUP-175(gap=0.357), SUP-179(gap=0.231)
