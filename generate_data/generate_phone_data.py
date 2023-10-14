import pandas as pd

def generate_phone_data(data):
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
    ls_14= []
    ls_15= []
    ls_16= []
    ls_17= []
    ls_18= []
    for i in data.iloc:
        cauhoi_1= f"sản phẩm {i.iloc[0]} có trang bị chuẩn IP68 không?"
        traloi_1 = f"không ạ, sản phẩm {i.iloc[0]} không trang bị chuẩn kháng nước IP68."

        cauhoi_2= f"sản phẩm {i.iloc[0]} phiên bản {i.iloc[3]} ram đang có giá bao nhiêu?"
        traloi_2= f"{i.iloc[0]} phiên bản {i.iloc[3]} ram đang có giá {i.iloc[1]}đ tại tất cả hệ thống của shop ạ."

        cauhoi_3= f"sản phẩm {i.iloc[0]} phiên bản {i.iloc[3]} đang có giá bao nhiêu?"
        traloi_3= f"{i.iloc[0]} phiên bản {i.iloc[3]}gb ram đang có giá {i.iloc[1]}."

        cauhoi_4= f"kích thước màn hình của {i.iloc[0]} bản {i.iloc[3]}gb ram là bao nhiêu?"
        traloi_4= f"kích thước màn hình của sản phẩm {i.iloc[0]} phiên bản {i.iloc[3]}gb ram là {i.iloc[4]} ạ."

        cauhoi_5= f"cấu hình của {i.iloc[0]} phiên bản {i.iloc[3]}gb?"
        traloi_5= f"{i.iloc[0]} phiên bản {i.iloc[3]}gb được trang bị chip {i.iloc[2]}, {i.iloc[3]}gb bộ nhớ trong, màn hình {i.iloc[4]}, 2 camera sau và 1 camera trước và độ phân giải mỗi camera là {i.iloc[5]}."

        cauhoi_6= f"{i.iloc[0]} hiện tại đang có giá bao nhiêu?"
        traloi_6= f"xin lỗi quý khách, bên shop không kinh doanh sản phẩm này."

        cauhoi_7= f"sản phẩm {i.iloc[0]} còn hàng không?"
        traloi_7= f"{i.iloc[0]} đang còn hàng ở tất cả đại lý shop ạ."

        cauhoi_8= f"sản phẩm {i.iloc[0]} còn hàng không?."
        traloi_8= f"hiện tại, sản phẩm {i.iloc[0]} đang tạm thời hết hàng. Vui lòng quay lại sau."

        cauhoi_9= f"tôi cần mua {i.iloc[0]}, bên hãng mình còn hàng không?"
        traloi_9= f"hiện tại {i.iloc[0]} chưa về hàng ạ."

        cauhoi_10= f"điện thoại {i.iloc[0]} hiện tại đang có giá bao nhiêu?"
        traloi_10= f"hiện tại sản phẩm {i.iloc[0]}đang có giá {i.iloc[1]}đ ạ."

        cauhoi_11= f"giá của sản phẩm {i.iloc[0]} khi về shop là bao nhiêu?"
        traloi_11= f"khi về Việt Nam, giá của sản phẩm {i.iloc[0]} có giá khoảng {i.iloc[1]}đ ạ."

        cauhoi_12= f"Cho em hỏi {i.iloc[0]} cũ {i.iloc[3]}gb khoảng bao nhiêu ạ?"
        traloi_12= f"Dạ sản phẩm này bên Shop hiện tại không kinh doanh ạ. Mình vui lòng tham khảo các sản phẩm khác tương tự nhé. Để được hỗ trợ chi tiết hơn về sản phẩm quý khách liên hệ tổng đài miễn phí 18006601 ạ."

        cauhoi_13= f"Shop mình có bán {i.iloc[0]} không?"
        traloi_13= f"Dạ có ạ, Shop mình có bán {i.iloc[0]} với mức giá {i.iloc[1]}đ."

        cauhoi_14= f"Điện thoại {i.iloc[0]} có hỗ trợ sạc nhanh không?"
        traloi_14= f"Dạ có ạ."

        cauhoi_15= f"Điện thoại {i.iloc[0]} có hỗ trợ sạc nhanh không?"
        traloi_15= f"Dạ không ạ."

        cauhoi_16= f"điện thoại {i.iloc[0]} giá bao nhiêu?"
        traloi_16= f"theo chương trình khuyến mãi, {i.iloc[0]} bản {i.iloc[3]}gb có giá là {i.iloc[1]}đ ạ."

        cauhoi_17= f"{i.iloc[0]} được trang bị chip nào?"
        traloi_17= f"{i.iloc[0]} được trang bị {i.iloc[2]} cực kì mạng mẽ."

        cauhoi_18= f"Cho tôi đặt hàng sản phẩm {i.iloc[0]}."
        traloi_18= f"Xin lỗi quý khách, sản phẩm hiện đang hết hàng ạ."

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
        ls_14.append([cauhoi_14, traloi_14])
        ls_15.append([cauhoi_15, traloi_15])
        ls_16.append([cauhoi_16, traloi_16])
        ls_17.append([cauhoi_17, traloi_17])
        ls_18.append([cauhoi_18, traloi_18])
    
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
    data14 = pd.DataFrame(ls_14,columns=["cauhoi","traloi"])
    data15 = pd.DataFrame(ls_15,columns=["cauhoi","traloi"])
    data16 = pd.DataFrame(ls_16,columns=["cauhoi","traloi"])
    data17 = pd.DataFrame(ls_17,columns=["cauhoi","traloi"])
    data18 = pd.DataFrame(ls_18,columns=["cauhoi","traloi"])

    data= pd.concat([data1,data2,data3,data4,data5,data6,data7,data8,
                     data9,data10,data11,data12,data13,data14,data15,data16,
                     data17,data18])

    return data