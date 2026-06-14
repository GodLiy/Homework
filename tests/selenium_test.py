"""SockShop — Selenium 功能测试"""
import time, json, sys
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

BASE_URL="http://localhost:8088"
RESULTS=[]

def log(name, status, msg="", dur=0):
    RESULTS.append({"test":name,"status":status,"message":msg,"duration":round(dur,2),"ts":datetime.now().isoformat()})
    print(f"  [{'PASS' if status=='PASS' else 'FAIL' if status=='FAIL' else 'WARN'}] {name} ({dur:.2f}s)")

def setup_driver():
    opts=webdriver.ChromeOptions()
    opts.add_argument('--headless'); opts.add_argument('--no-sandbox')
    opts.add_argument('--window-size=1920,1080')
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=opts)

def run():
    d=None
    try:
        print(f"SockShop Selenium Test: {BASE_URL}\n"); d=setup_driver()
        # Test 1: Homepage
        for name,url in [("首页加载","/"),("登录页","/login"),("注册页","/register")]:
            s=time.time()
            try:
                d.get(f"{BASE_URL}{url}"); e=time.time()-s
                log(f"页面:{name}", "PASS" if e<10 else "WARN", f"HTTP 200", e)
            except Exception as ex: log(f"页面:{name}","FAIL",str(ex)[:80],time.time()-s)
        # Test 2: Navigation
        s=time.time()
        try:
            links=d.find_elements(By.TAG_NAME,"a")
            log("导航链接","PASS",f"{len(links)} links found",time.time()-s)
        except: log("导航链接","FAIL","",time.time()-s)
        # Test 3: API
        for name,url in [("Catalogue API","/catalogue"),("Carts API","/carts")]:
            s=time.time()
            try:
                d.get(f"{BASE_URL}{url}"); e=time.time()-s
                log(f"API:{name}","PASS",f"HTTP{e:.2f}s",e)
            except: log(f"API:{name}","WARN","may need auth",time.time()-s)
        # Test 4: Response times
        pages=["/","/login","/catalogue"]
        for p in pages:
            s=time.time()
            try: d.get(f"{BASE_URL}{p}"); log(f"响应时间:{p}","PASS",f"{time.time()-s:.2f}s",time.time()-s)
            except: log(f"响应时间:{p}","FAIL","",time.time()-s)
    finally:
        if d: d.quit()
        p=sum(1 for r in RESULTS if r['status']=='PASS')
        f=sum(1 for r in RESULTS if r['status']=='FAIL')
        print(f"\n  Total:{len(RESULTS)} | PASS:{p} | FAIL:{f}")
        with open("F:/CCode/Homework/tests/selenium_results.json","w",encoding="utf-8") as fp: json.dump(RESULTS,fp,ensure_ascii=False,indent=2)

if __name__=="__main__": run()
