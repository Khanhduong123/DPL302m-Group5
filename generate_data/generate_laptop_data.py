import pandas as pd


def generate_laptop_data(data):
    ls_1 = []   
    ls_2= []
    ls_3= []
    ls_4= []
    ls_5= []
    ls_6= []
    ls_7= []
    ls_8= []
    ls_9= []
    ls_10= []
    ls_11= []
    ls_12= []
    ls_13= []
    for i in data.iloc:
        cauhoi_1 = f"Giá của {i.iloc[0]}?"
        traloi_1= f"Xin lỗi quý khách, hiện tại shop mình không kinh doanh sản phẩm này ạ."

        cauhoi_2= f"{i.iloc[0]} có giá bao nhiêu?"
        traloi_2= f"Hiện tại, {i.iloc[0]} có giá là {i.iloc[1]}đ."

        cauhoi_3= f"Giá của {i.iloc[0]}?"
        traloi_3= f"{i.iloc[0]} có giá là {i.iloc[1]}đ."

        cauhoi_4=  f"Shop mình có bán {i.iloc[0]} không?"
        traloi_4= f"Dạ có ạ, Shop mình có bán {i.iloc[0]} với mức giá {i.iloc[1]}đ."

        cauhoi_5= f"{i.iloc[0]} hiện tại đang có giá bao nhiêu?"
        traloi_5= f"theo chương trình khuyến mãi, {i.iloc[0]} bản {i.iloc[3]}gb ram có giá là {i.iloc[1]}đ ạ."

        cauhoi_6= f"giá của sản phẩm {i.iloc[0]} khi về shop là bao nhiêu?"
        traloi_6= f"khi về Việt Nam, giá của sản phẩm {i.iloc[0]} có giá khoảng {i.iloc[1]}đ ạ."

        cauhoi_7= f"{i.iloc[0]} được trang bị chip nào?"
        traloi_7= f"{i.iloc[0]} được trang bị {i.iloc[2]}."

        cauhoi_8= f"{i.iloc[0]} được trang bị card đồ hoạ nào?"
        traloi_8= f"{i.iloc[0]} được trang bị {i.iloc[5]}."

        cauhoi_9= f"{i.iloc[0]} có dung lượng ổ cứng là bao nhiêu gb?"
        traloi_9= f"sản phẩm {i.iloc[0]} được trang bị ổ cứng {i.iloc[4]} gb."

        cauhoi_10= f"cấu hình của {i.iloc[0]}?"
        traloi_10= f"{i.iloc[0]} được trang bị chip {i.iloc[2]} ổ cứng có dung lượng {i.iloc[4]} gb,và có 1 card màn {i.iloc[5]} được tích hợp trong máy."

        cauhoi_11= f"sản phẩm {i.iloc[0]} phiên bản ram là {i.iloc[3]} đang có giá bao nhiêu?"
        traloi_11= f"{i.iloc[0]} phiên bản bản ram là {i.iloc[3]} đang có giá {i.iloc[1]}đ tại tất cả hệ thống của shop ạ."

        cauhoi_12= f"tôi cần mua {i.iloc[0]}, bên hãng mình còn hàng không?"
        traloi_12= f"hiện tại {i.iloc[0]} chưa về hàng ạ."

        cauhoi_13= f"Cho em hỏi {i.iloc[0]} cũ khoảng bao nhiêu ạ?"
        traloi_13= f"Dạ sản phẩm này bên Shop hiện tại không kinh doanh ạ. Mình vui lòng tham khảo các sản phẩm khác tương tự nhé. Để được hỗ trợ chi tiết hơn về sản phẩm quý khách liên hệ tổng đài miễn phí 18006601 ạ."

        ls_1.append([cauhoi_1, traloi_1])      
        ls_2.append([cauhoi_2, traloi_2])       
        ls_3.append([cauhoi_3, traloi_3])       
        ls_4.append([cauhoi_4, traloi_4])       
        ls_5.append([cauhoi_5, traloi_5])       
        ls_6.append([cauhoi_6, traloi_6])       
        ls_7.append([cauhoi_7, traloi_7])       
        ls_8.append([cauhoi_8, traloi_8])       
        ls_9.append([cauhoi_9, traloi_9])       
        ls_10.append([cauhoi_10, traloi_10])
        ls_11.append([cauhoi_11, traloi_11])
        ls_12.append([cauhoi_12, traloi_12])
        ls_13.append([cauhoi_13, traloi_13])

    data1 = pd.DataFrame(ls_1,columns=["cauhoi","traloi"])
    data2 = pd.DataFrame(ls_2,columns=["cauhoi","traloi"])
    data3 = pd.DataFrame(ls_3,columns=["cauhoi","traloi"])
    data4 = pd.DataFrame(ls_4,columns=["cauhoi","traloi"])
    data5 = pd.DataFrame(ls_5,columns=["cauhoi","traloi"])
    data6 = pd.DataFrame(ls_6,columns=["cauhoi","traloi"])
    data7 = pd.DataFrame(ls_7,columns=["cauhoi","traloi"])
    data8 = pd.DataFrame(ls_8,columns=["cauhoi","traloi"])
    data9 = pd.DataFrame(ls_9,columns=["cauhoi","traloi"])
    data10 = pd.DataFrame(ls_10,columns=["cauhoi","traloi"])
    data11 = pd.DataFrame(ls_11,columns=["cauhoi","traloi"])
    data12 = pd.DataFrame(ls_12,columns=["cauhoi","traloi"])
    data13 = pd.DataFrame(ls_13,columns=["cauhoi","traloi"])

    data= pd.concat([data1,data2,data3,data4,data5,data6,data7,data8,
                     data9,data10,data11,data12,data13])

    return data

