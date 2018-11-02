from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import time


city_urls = """http://epaper.bhaskar.com/'jaipur/14/DATEHERE/0/1/
http://epaper.bhaskar.com/'jodhpur/17/DATEHERE/0/1/
http://epaper.bhaskar.com/'pali/40/DATEHERE/0/1/
http://epaper.bhaskar.com/'udaipur/18/DATEHERE/0/1/
http://epaper.bhaskar.com/'ajmer/255/DATEHERE/0/1/
http://epaper.bhaskar.com/'kota/16/DATEHERE/0/1/
http://epaper.bhaskar.com/'ganganagar/41/DATEHERE/0/1/
http://epaper.bhaskar.com/'dausabhaskar/426/DATEHERE/0/1/
http://epaper.bhaskar.com/'bhilwara/158/DATEHERE/0/1/
http://epaper.bhaskar.com/'sikar/154/DATEHERE/0/1/
http://epaper.bhaskar.com/'alwar/36/DATEHERE/0/1/
http://epaper.bhaskar.com/'bikaner/191/DATEHERE/0/1/
http://epaper.bhaskar.com/'sawaimadhopur/35/DATEHERE/0/1/
http://epaper.bhaskar.com/'tonk/85/DATEHERE/0/1/
http://epaper.bhaskar.com/'dausa/86/DATEHERE/0/1/
http://epaper.bhaskar.com/'karauli/144/DATEHERE/0/1/
http://epaper.bhaskar.com/'balotra/145/DATEHERE/0/1/
http://epaper.bhaskar.com/'badmer/146/DATEHERE/0/1/
http://epaper.bhaskar.com/'jaisalmer/147/DATEHERE/0/1/
http://epaper.bhaskar.com/'nagour/148/DATEHERE/0/1/
http://epaper.bhaskar.com/'jalore/92/DATEHERE/0/1/
http://epaper.bhaskar.com/'sirohi/93/DATEHERE/0/1/
http://epaper.bhaskar.com/'dungarpur/90/DATEHERE/0/1/
http://epaper.bhaskar.com/'rajsamabandh/91/DATEHERE/0/1/
http://epaper.bhaskar.com/'banswara/161/DATEHERE/0/1/
http://epaper.bhaskar.com/'baran/94/DATEHERE/0/1/
http://epaper.bhaskar.com/'bundi/95/DATEHERE/0/1/
http://epaper.bhaskar.com/'churu/155/DATEHERE/0/1/
http://epaper.bhaskar.com/'jhalawar/96/DATEHERE/0/1/
http://epaper.bhaskar.com/'hanumangarh/149/DATEHERE/0/1/
http://epaper.bhaskar.com/'chittorgarh/199/DATEHERE/0/1/
http://epaper.bhaskar.com/'jhunjhunu/156/DATEHERE/0/1/
http://epaper.bhaskar.com/'dholpur/151/DATEHERE/0/1/
http://epaper.bhaskar.com/'gangapurcitybhaskar/455/DATEHERE/0/1/ 
http://epaper.bhaskar.com/'bharatpur/152/DATEHERE/0/1/
http://epaper.bhaskar.com/'sumerpursheogange/408/DATEHERE/0/1/
http://epaper.bhaskar.com/'mountabuaburoad/407/DATEHERE/0/1/
http://epaper.bhaskar.com/'shimlabhaskar/184/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'solanbhaskar/185/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'himachaledition/186/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'jalandhar/56/DATEHERE/cph/1/
http://epaper.bhaskar.com/'amritsar/58/DATEHERE/cph/1/
http://epaper.bhaskar.com/'ludhiana/63/DATEHERE/cph/1/
http://epaper.bhaskar.com/'bathinda/181/DATEHERE/cph/1/
http://epaper.bhaskar.com/'patiala/182/DATEHERE/cph/1/
http://epaper.bhaskar.com/'hoshiyarpur/65/DATEHERE/cph/1/
http://epaper.bhaskar.com/'navansaharbhaskar/66/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'kapurthalabhaskar/67/DATEHERE/cph/1/
http://epaper.bhaskar.com/'pathankotbhaskar/69/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'gurdaspurbhaskar/70/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'roparbhaskar/294/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'firozpurbhaskar/295/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'mogabhaskar/296/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'sangroorbhaskar/297/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'fatehgarhbhaskar/298/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'fazilkapullout/396/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'mansabhaskar/514/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'panipat/60/DATEHERE/cph/1/
http://epaper.bhaskar.com/'hisar/97/DATEHERE/cph/1/
http://epaper.bhaskar.com/'karnalbhaskar/102/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'sonipatbhaskar/104/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'ambalabhaskar/105/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'faridabadbhaskar/106/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'gurgaonbhaskar/107/DATEHERE/cph/1/   
http://epaper.bhaskar.com/'rohtakbhaskar/103/DATEHERE/cph/1/   
http://epaper.bhaskar.com/'jindbhaskar/143/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'sirsabhaskar/98/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'rewaribhaskar/99/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'fatehabadbhaskar/100/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'bhiwanibhaskar/101/DATEHERE/cph/1/     
http://epaper.bhaskar.com/'yamunanagarbhaskar/139/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'kurukshetrabhaskar/140/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'kaithalbhaskar/141/DATEHERE/cph/1/
http://epaper.bhaskar.com/'narnaulbhaskar/142/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'jhajjarbhaskar/265/DATEHERE/cph/1/  
http://epaper.bhaskar.com/'gohanapullout/389/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'hansipullout/390/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'dadripullout/391/DATEHERE/cph/1/  
http://epaper.bhaskar.com/'dabawalipullout/392/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'bahadurgarh/398/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'mahendergarhpullout/403/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'chandigarhcity/193/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'mohalibhaskar/319/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'citylife/266/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'zirakpurbhaskar/535/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'chandigarhbhaskar/83/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'panchkulabhaskar/267/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'bhopal/120/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'itarsi/126/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'indore/129/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'ratlam/132/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'gwalior/135/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'ujjain/164/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'sagar/167/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'shivpuri pullout/359/DATEHERE/mpcg/1/ 
http://epaper.bhaskar.com/'vidisha/121/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'sheopur bhaskar/380/DATEHERE/mpcg/1/ 
http://epaper.bhaskar.com/'sehore/122/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'guna/123/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'harda/128/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'jhabua/130/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'khandwa/162/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'khargone/163/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'neemuch/133/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'mandsour/134/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'bhind/171/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'shajapur/165/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'chattarpur/168/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'damoh/169/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'raisan/269/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'rajgarh/270/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'nagda/166/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'badwani/275/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'dhar/276/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'morena/277/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'devasbhaskar/375/DATEHERE/mpcg/1/ 
http://epaper.bhaskar.com/'mhowbhaskar/376/DATEHERE/mpcg/1/ 
http://epaper.bhaskar.com/'datiabhaskar/381/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'tikamgarh/400/DATEHERE/mpcg/1/ 
http://epaper.bhaskar.com/'binabhaskar/451/DATEHERE/mpcg/1/ 
http://epaper.bhaskar.com/'dabrabhaskar/453/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'raipur/116/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'bhilai/119/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'bilaspur/172/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'raigarh/268/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'rajnandgaon/117/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'bastar/118/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'dhamtari/176/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'mahasamund/177/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'anchalik/178/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'jashpur/173/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'korba/175/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'janjgir/174/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'koria/280/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'ambikapur/278/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'kanker/279/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'kawardhabhaskar/466/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'bilaspuranchlik/472/DATEHERE/mpcg/1/ 
http://epaper.bhaskar.com/'newdelhi/194/DATEHERE/cph/1/ 
http://epaper.bhaskar.com/'suratcityhindi/488/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'ranchi/109/DATEHERE/0/1/
http://epaper.bhaskar.com/'jamshedpur/195/DATEHERE/0/1/
http://epaper.bhaskar.com/'dhanbad/221/DATEHERE/0/1/
http://epaper.bhaskar.com/'ramgarh/110/DATEHERE/0/1/
http://epaper.bhaskar.com/'palamu/111/DATEHERE/0/1/
http://epaper.bhaskar.com/'koderma/112/DATEHERE/0/1/
http://epaper.bhaskar.com/'hazaribagh/113/DATEHERE/0/1/
http://epaper.bhaskar.com/'gumlasimdega/114/DATEHERE/0/1/
http://epaper.bhaskar.com/'garhwa/486/DATEHERE/0/1/
http://epaper.bhaskar.com/'giridihbhaskar/224/DATEHERE/0/1/
http://epaper.bhaskar.com/'chaibasa/219/DATEHERE/0/1/
http://epaper.bhaskar.com/'jamtarabhaskar/223/DATEHERE/0/1/
http://epaper.bhaskar.com/'ghatsila/220/DATEHERE/0/1/
http://epaper.bhaskar.com/'bokaro/248/DATEHERE/0/1/
http://epaper.bhaskar.com/'jharkhandspecial/260/DATEHERE/0/1/
http://epaper.bhaskar.com/'patnacity/384/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'bhagalpurcity/456/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'gaya/521/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'bettiah/522/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'muzaffarpurbhaskar/458/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'hajipurbhaskar/459/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'biharsharifbhaskar/460/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'arabhaskar/461/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'chhaprabhaskar/462/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'purnea/463/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'samastipur/464/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'darbhanga/465/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'aurangabad/474/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'sasaram/475/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'gopalganj/476/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'siwan/477/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'buxar/478/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'nawada/479/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'begusarai/480/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'madhubani/481/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'motihari/482/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'katihar/483/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'jahanabadbhaskar/494/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'bhabhuabhaskar/495/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'arariabhaskar/496/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'saharsabhaskar/497/DATEHERE/bihar/1/ 
http://epaper.bhaskar.com/'supaulbhaskar/498/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'madhepurabhaskar/499/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'khagariabhaskar/500/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'jamuibhaskar/501/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'lakhisaraibhaskar/502/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'mungerbhaskar/503/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'bankabhaskar/504/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'sitamarhibhaskar/506/DATEHERE/bihar/1/
http://epaper.bhaskar.com/'sheikhpura bhaskar/515/DATEHERE/bihar/1/ 
http://epaper.bhaskar.com/'kaimur bhaskar/516/DATEHERE/bihar/1/ 
http://epaper.bhaskar.com/'kishanganjbhaskar/519/DATEHERE/bihar/1/ 
http://epaper.bhaskar.com/'baghabhaskar/531/DATEHERE/bihar/1/ 
http://epaper.bhaskar.com/'magazine/ahazindagi/211/DATEHERE/mpcg/1/ 
http://epaper.bhaskar.com/'rasrangpunjab/449/DATEHERE/mpcg/1/ 
http://epaper.bhaskar.com/'navrangpunjab/450/DATEHERE/mpcg/1/ 
http://epaper.bhaskar.com/'magazine/navrang/212/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'magazine/navrangbollywood/299/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'magazine/madhurima/213/DATEHERE/mpcg/1/
http://epaper.bhaskar.com/'yugalbhaskar/509/DATEHERE/mpcg/1/ 
http://epaper.bhaskar.com/'magazine/balbhaskar/214/DATEHERE/mpcg/1/ 
http://epaper.bhaskar.com/'magazine/young bhaskar/216/DATEHERE/mpcg/1/"""

city_urls = city_urls.split("\n")
dates = [str(i).zfill(2) + "072018" for i in range(1,31)]
urls = []
for city_url in city_urls:
    for date in dates:
        urls.append(city_url.replace("DATEHERE", date))
        

driver = webdriver.Firefox()

output = []
for url in urls:
    driver.get(url)
    pageDropDown = driver.find_element_by_name("page")
    select = Select(pageDropDown)
    for page_no  in select.options:
        page_no.click()
        time.sleep(1)
        pdfLink = driver.find_element_by_id("pdfLink")
        a = pdfLink.find_element_by_tag_name("a")
        pdf_url = a.get_attribute("href")
        arts = driver.find_elements_by_class_name("borderimage")
        coords = []
        for art in arts:
            coords.append(map(float, art.get_attribute("coords").split(",")))
        output.append([pdf_url, coords])
    #break
    
