#!/usr/bin/env python3
"""
Warm-up script for Cloud Run service
Sends gradual requests to ensure all instances are ready
"""

import asyncio
import aiohttp
import time

SERVICE_URL = "https://mannequin-segmenter-new-234382015820.europe-west4.run.app/infer"
IMAGE_URL = "https://ff81be21ba9ff5a7e3e96982d1d81ee9b201ced3fece7b9cc9cea70-apidata.googleusercontent.com/download/storage/v1/b/public-images-redi/o/0_Damski-pulover-Public-130242404a.webp?jk=AXbWWmn7roAtULCgUkK_DVAe7cty-hpaZqq9g2qBj6Pz8g70rLdpAMPGb5HG0TDGrRCCcJdX2P7b8Ofcz6nr0bgxOQSZ4miKbiCCY9IXt0oclyVDmujhKsgQGGBb6ns_EcBELESVhbqG7obgvbovnyfrt5x5fIGUXftFqBC13JK_Vb3Eth300MvKYFL25xjkiBhJmzpj4zmxM1BDZ-mZ4EARYZZgbnkqax9Rzep7uxyIfcKb1WJu4Xha6dLjGXJOFssPQdPN6N6aVvrPKPHbVJwnG0F8SozdZcqtvnY928kANUyDPXUaxwRUkuHP7oX5qBabOa6k8BewU_DgXYqGXYD2hC6zDzgHTo6dILaUuU8g2l5luk2wHOP1OC780zFE6wlGr2f5iNQrJSpO78KxtNL2qJvG-kQS7vUdKziSKmzdTocR7yzlPeLi9ns7q7UhBHIutsOvtUAwIQIaQGfvkB5c1lXBnFXL_Kj2RVwN6E50DXNFZ8BBHIPBS9cgXz6fXkICGIdbIfoTeTHNrcASnmqpEmGZMJaYIq0THv-3kkrDOFRJOlw1UqrTzerz2wC2yajZgKtv0fhndf5QYTdf-kvXudoGbi4xWVlQf2eCSqxwxFVo2r-vx2Ke8NHnvrWd6HdTPVbasSVBZih5knjhX6CLwAuDOsNSjM96I7RoN2f6Fc12JoKisapnT2u7yxiAcsIVjECfJGLdtwWt1kJtgDTBGlQ73h9eXlWeE3b_SHtP7fU5HE5Sstj33uvkErnl3CqJUPh4-TccsgYK__PhsCmXuoBqxkz_8UTVaNblSD2hYVN10sOb29dp7CosGcMz79YxbhKyQ0vKQESkSNnj4DBKId9rkoFNK01fB0omzlftCneqMmj2ZftqDLKE87Vly2c0jmwPnJT8RsKvy29rDuAezLmTTYqWSf8DdlsH1aUBKtKUv48Mb-Y069RXHDbm9i-xlDejDQc3_Qz1xzZD8-YWaQvm1TN8uJk8SU0RTuhwpybM-eoDTyeroVxWPsKeoe9SCNvYVu8DE7upPr4sVUSvG-yZXFFUdaHUg_u1ouKpNgJa3a194OHduYFqhpgu_CoA1wY-vgVBeIJ96KPUTudVhesTVtHDtnzWCAmOpHRpkmrMgbWGUPygGI29T6dq0OKF_sjyCLgbrnd_3-cIUaf-1IFwSitQ56Mc4L3vWlH-YLlgW5qgvCvToMEfaHKbKl-Ivewm7GRUV4oIEAxwAKr_az-i5J47YgPdp4ioFlakfhkKERfORXox5wHLuGRTE4AxAfXU0WDCkOEFXmstZ1bl9WYBj-T3N6azirnjFSvcLxV4_il_PofJxJFZe1bNtPp3Qkrin3FD2LeuFMQilmELyMFrwnUDUcflmw&isca=1"

async def warmup_request(session, request_id):
    """Send a single warm-up request"""
    payload = {"image_url": IMAGE_URL, "upload_gcs": True}
    
    try:
        start_time = time.time()
        async with session.post(SERVICE_URL, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
            duration = time.time() - start_time
            status = response.status
            print(f"Warmup {request_id:2d}: {duration:5.2f}s (HTTP {status})")
            return status == 200
    except Exception as e:
        print(f"Warmup {request_id:2d}: ERROR - {e}")
        return False

async def main():
    print("🔥 WARMING UP CLOUD RUN SERVICE")
    print("=" * 50)
    print(f"Target: {SERVICE_URL}")
    print("Strategy: Gradual ramp-up to activate all instances")
    print()
    
    connector = aiohttp.TCPConnector(limit=50, limit_per_host=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        
        # Phase 1: Single requests to wake up min-instances
        print("Phase 1: Wake up instances (5 sequential requests)")
        for i in range(5):
            await warmup_request(session, i + 1)
            await asyncio.sleep(2)  # 2s between requests
        
        print()
        
        # Phase 2: Gradual concurrent increase  
        print("Phase 2: Gradual ramp-up (activate auto-scaling)")
        concurrency_levels = [2, 5, 10, 15]
        
        for level in concurrency_levels:
            print(f"Testing {level} concurrent requests...")
            
            tasks = []
            for i in range(level):
                task = warmup_request(session, i + 1)
                tasks.append(task)
                await asyncio.sleep(0.1)  # Small delay between starts
            
            results = await asyncio.gather(*tasks)
            success_rate = sum(results) / len(results) * 100
            print(f"  ✅ Success rate: {success_rate:.1f}% ({sum(results)}/{len(results)})")
            
            await asyncio.sleep(3)  # Wait between phases
        
        print()
        print("🎉 Warm-up completed! Service should be ready for load testing.")

if __name__ == "__main__":
    asyncio.run(main())
