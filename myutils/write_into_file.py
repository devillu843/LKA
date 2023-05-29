import pandas as pd


def pd_toExcel(img, id1, p1, id2, p2, id3, p3, fileName):  # pandas库储存数据到excel

    dfData = {  # 用字典设置DataFrame所需数据
        '图片名称': img,
        '第一预测种类': id1,
        '第一预测可能性': p1,
        '第二预测种类': id2,
        '第二预测可能性': p2,
        '第三预测种类': id3,
        '第三预测可能性': p3,

    }
    df = pd.DataFrame(dfData)  # 创建DataFrame
    df.to_excel(fileName, index=False)  # 存表，去除原始索引列（0,1,2...）
