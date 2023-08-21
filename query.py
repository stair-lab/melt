# import os
# import openai
# # openai.organization = "org-UpulfpTkmLGB4m3pTF3YPGrg"
# openai.organization = "org-pIlJPPZw4ljlb4pOExU3DfHR"
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.Model.list()

# completion = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello!"}
#   ]
# )

# print(completion.choices[0].message)



## BARD
import requests
from bardapi import Bard, BardCookies, SESSION_HEADERS

session = requests.Session()
session.cookies.set("__Secure-1PSID", "ZwjDWj_mFy0ARuT3_mhQcpBop4jxG7GlYHkpb3FRUVUDV29cjhnbxAaVlrxIxxHeqWb9uA.")
session.cookies.set( "__Secure-1PSIDCC", "APoG2W_G0PFthXAi8x7efqZaKGsY3zatF-ST3w35dJQ7SvW2Dfi_233FGafe93mhkEyXCWcHQ4M")
session.cookies.set("__Secure-1PSIDTS", "sidts-CjIBSAxbGYayZcmjVGffAuluYn-3IIqPl6sCWMOlv20MfYmC7gXwa37tuVIigGn59YCYYhAA")
session.headers = SESSION_HEADERS

bard = Bard(session=session, token="ZwjDWj_mFy0ARuT3_mhQcpBop4jxG7GlYHkpb3FRUVUDV29cjhnbxAaVlrxIxxHeqWb9uA.")

ans = bard.get_answer("Bạn là một trợ lý thông minh. Nhiệm vụ của bạn là đứng trên góc độ của một con người "
                      "và đưa ra sự lựa chọn mà bạn cho là đúng đắn và hợp lý hơn. "
                      "Bạn chỉ cần trả lời là ``Lựa chọn 1`` hoặc ``Lựa chọn 2``.\n\n"
                      "Câu hỏi: Đội thủ Panthers đã thua bao nhiêu điểm?\n"
                      "Ngữ cảnh: Đội thủ của Panthers chỉ thua 308 điểm, đứng thứ sáu trong giải đấu, đồng thời dẫn đầu NFL về số lần đoạt bóng (intercept) với 24 lần và tự hào với bốn lựa chọn Pro Bowl. Người húc (tackle) trong đội thủ tham gia Pro Bowl, Kawann Short dẫn đầu đội về số lần vật ngã (sack) với 11 lần, đồng thời húc văng bóng (fumble) 3 lần và lấy lại được bóng (recover) 2 lần. Đồng nghiệp lineman Mario Addison đã thêm 6½ lần vật ngã. Đội hình Panthers cũng có người tiền vệ (defensive end) kỳ cựu Jared Allen, người tham gia Pro Bowl 5 lần, dẫn đầu về số lần vật ngã trong sự nghiệp NFL với 136 lần, cùng với người tiền vệ Kony Ealy, người đã có 5 lần vật ngã sau 9 lần xuất phát. Phía sau họ, hai trong số ba người hàng vệ (linebacker) xuất phát của Panthers cũng được chọn để chơi trong Pro Bowl: Thomas Davis và Luke Kuechly. Davis đã có 5½ lần vật ngã, 4 lần húc văng bóng và 4 lần đoạt bóng, trong khi Kuechly dẫn đầu đội về số lần húc (118 lần), 2 lần húc văng bóng và đoạt bóng từ các đường chuyền 4 lần. Đội hình phía sau của Carolina có hậu-hậu vệ (safety) tham gia Pro Bowl Kurt Coleman, người dẫn đầu đội bóng với bảy lần đoạt bóng trong sự nghiệp, đồng thời có 88 cú húc bóng và trung vệ (cornerback) tham gia Pro Bowl Josh Norman, người đã phát triển thành một shutdown corner trong mùa giải và có bốn lần đoạt bóng, hai trong số đó đã trở thành touchdown.\n"
                      "Lựa chọn 1: Đội thủ Panthers đã thua 308 điểm.\n"
                      "Lựa chọn 2: 308\n\n"
                      "Lựa chọn của bạn: ")['content']

print(ans)
