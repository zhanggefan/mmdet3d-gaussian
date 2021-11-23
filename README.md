PointPillars On KITTI
===

3D
---

|Model|tau|fun|Car/Easy|Car/Mod|Car/Hard|Ped/Easy|Ped/Mod|Ped/Hard|Cyc/Easy|Cyc/Mod|Cyc/Hard|Ovr/Easy|Ovr/Mod|Ovr/Hard|download|
|-----|---|---|--------|-------|--------|--------|-------|--------|--------|-------|--------|--------|-------|--------|--------|
|baseline|-|-|86.3442|76.8062|73.5068|57.1734|52.6698|48.0374|81.6086|65.5086|61.4947|75.0421|64.9949|61.0130|[log](https://drive.google.com/file/d/1ODZk8Mr3ECR7gHvjCEl5iemWD-6sxwd-/view?usp=sharing)  [weight](https://drive.google.com/file/d/1T0xWOU7tZUJGERwAotb-zrKYReUdBI0a/view?usp=sharing)|
|gwd|0|f(x)=x|86.4740|76.6016|75.1838|58.9054|54.3385|49.8206|83.5646|65.9063|62.1912|76.3147|65.6154|62.3985|[log](https://drive.google.com/file/d/148W3FrDgvMKKCi0Y6DdsCfcu8IcUhMc6/view?usp=sharing)  [weight](https://drive.google.com/file/d/1WjAmYIDIw1ZoSMRvNxgxACLixW5mc1sg/view?usp=sharing)|
|gwd|0|f(x)=log(1+x)|86.1492|76.7271|75.1031|**61.0951**|**55.2907**|**50.8962**|82.8355|65.9086|62.3309|76.6933|65.9755|62.7767|[log](https://drive.google.com/file/d/17rucClwvUlLr2AMj7liJhBUUz-VWIPHg/view?usp=sharing)  [weight](https://drive.google.com/file/d/1dW-DNy9wct3N_52JstPDRYP5RQYuryHX/view?usp=sharing)|
|kld|0|f(x)=x|87.4018|77.2049|75.9145|58.2495|52.5324|47.8418|81.2845|66.1001|61.4003|75.6453|65.2791|61.7189|[log](https://drive.google.com/file/d/1JUxvZuAxy-QOE-UpuTlxfzhPQOrW0767/view?usp=sharing)  [weight](https://drive.google.com/file/d/1dPh79KU0UIlKgaw9NbsIZtXJCqUbm39a/view?usp=sharing)|
|kld|0|f(x)=log(1+x)|**87.7164**|77.5651|75.6711|59.9753|54.3827|50.0470|82.4651|66.2440|62.7426|76.7189|**66.0640**|**62.8202**|[log](https://drive.google.com/file/d/1szkMl0bdX1fzXimnxiW64EZjOhe6NmmU/view?usp=sharing)  [weight](https://drive.google.com/file/d/1rKPwg2pk32TqtqU-emixpqoJ2gxpMS0S/view?usp=sharing)|
|bcd|0|f(x)=x|87.3245|**77.8775**|**76.4542**|58.7822|52.0467|47.7859|**87.8213**|**67.6561**|**63.7225**|**77.9760**|65.8601|62.6542|[log](https://drive.google.com/file/d/1fpfbPCGFtrvZddiJ0BTZ5LxqxgUeVKaE/view?usp=sharing)  [weight](https://drive.google.com/file/d/1M7CBfLs4uZ8eSpwokcXtH_GBwHNHBBIs/view?usp=sharing)|
|bcd|0|f(x)=log(1+x)|87.2366|76.9278|74.8263|57.3766|53.4922|49.3513|83.2023|66.6907|63.4531|75.9385|65.7036|62.5435|[log](https://drive.google.com/file/d/1FXqFqSEDGzQh8h6PO10lWAWDn5MleM1B/view?usp=sharing)  [weight](https://drive.google.com/file/d/1ZTmeWbjtHh0GonswMZ-d_cA9MeDMlZ9R/view?usp=sharing)|

BEV
---

|Model|tau|fun|Car/Easy|Car/Mod|Car/Hard|Ped/Easy|Ped/Mod|Ped/Hard|Cyc/Easy|Cyc/Mod|Cyc/Hard|Ovr/Easy|Ovr/Mod|Ovr/Hard|download|
|-----|---|---|--------|-------|--------|--------|-------|--------|--------|-------|--------|--------|-------|--------|--------|
|baseline|-|-|89.4991|86.6383|83.1225|62.4799|57.4521|53.1716|84.0316|68.5454|64.4239|78.6702|70.8786|66.9060|[log](https://drive.google.com/file/d/1ODZk8Mr3ECR7gHvjCEl5iemWD-6sxwd-/view?usp=sharing)  [weight](https://drive.google.com/file/d/1T0xWOU7tZUJGERwAotb-zrKYReUdBI0a/view?usp=sharing)|
|gwd|0|f(x)=x|89.1938|86.9307|85.3496|62.5435|57.2883|53.6182|85.3651|68.8068|64.7492|79.0341|71.0086|67.9056|[log](https://drive.google.com/file/d/148W3FrDgvMKKCi0Y6DdsCfcu8IcUhMc6/view?usp=sharing)  [weight](https://drive.google.com/file/d/1WjAmYIDIw1ZoSMRvNxgxACLixW5mc1sg/view?usp=sharing)|
|gwd|0|f(x)=log(1+x)|89.4604|87.1875|84.6354|**65.0736**|**59.4175**|**55.2898**|86.5754|70.8361|66.3322|80.3698|**72.4804**|**68.7525**|[log](https://drive.google.com/file/d/17rucClwvUlLr2AMj7liJhBUUz-VWIPHg/view?usp=sharing)  [weight](https://drive.google.com/file/d/1dW-DNy9wct3N_52JstPDRYP5RQYuryHX/view?usp=sharing)|
|kld|0|f(x)=x|**89.8162**|87.2874|**85.5490**|62.0096|55.6291|52.0079|86.3622|68.9297|64.8103|79.3960|70.6154|67.4557|[log](https://drive.google.com/file/d/1JUxvZuAxy-QOE-UpuTlxfzhPQOrW0767/view?usp=sharing)  [weight](https://drive.google.com/file/d/1dPh79KU0UIlKgaw9NbsIZtXJCqUbm39a/view?usp=sharing)|
|kld|0|f(x)=log(1+x)|89.7779|**87.5629**|83.7490|63.9494|57.6823|54.6880|86.5358|69.5444|66.2115|80.0877|71.5965|68.2162|[log](https://drive.google.com/file/d/1szkMl0bdX1fzXimnxiW64EZjOhe6NmmU/view?usp=sharing)  [weight](https://drive.google.com/file/d/1rKPwg2pk32TqtqU-emixpqoJ2gxpMS0S/view?usp=sharing)|
|bcd|0|f(x)=x|89.5253|87.4416|85.2996|63.8959|56.9703|52.7919|**88.8584**|**71.1039**|**66.6938**|**80.7598**|71.8386|68.2618|[log](https://drive.google.com/file/d/1fpfbPCGFtrvZddiJ0BTZ5LxqxgUeVKaE/view?usp=sharing)  [weight](https://drive.google.com/file/d/1M7CBfLs4uZ8eSpwokcXtH_GBwHNHBBIs/view?usp=sharing)|
|bcd|0|f(x)=log(1+x)|89.5482|86.6646|83.9405|62.9139|58.0344|54.6862|85.6345|69.1394|65.4903|79.3656|71.2794|68.0390|[log](https://drive.google.com/file/d/1FXqFqSEDGzQh8h6PO10lWAWDn5MleM1B/view?usp=sharing)  [weight](https://drive.google.com/file/d/1ZTmeWbjtHh0GonswMZ-d_cA9MeDMlZ9R/view?usp=sharing)|
