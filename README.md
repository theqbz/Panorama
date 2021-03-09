# Panorama
###### Panorámakép-készítő program - házi dolgozat Gépi látás (GKLB_INTM038) tárgyból.<br/>Készíti: Borsodi Zoltán (B7PK8Z), mérnökinformatika, Széchenyi István Egyetem

A program Python programozási nyelven készült az OpenCV függvénykönyvtár felhasználásával. Célja, hogy két fotóból panorámaképet készítsen.

## Mi az a panorámakép?

A panorámakép, több egymást átfedő, azonos forgatású képből előállított egyetlen – immár széles – kép. A több kép egyesítésével történő panorámakép alkotást összefűzésnek nevezzük. Az összefűzéshez olyan fotókból kell kiindulni, amelyek azonos állásúak és egymást valamilyen mértékben átfedik. A végeredmény egy egyesített kép lesz, ezeknek az átfedésben lévő fotóknak a felhasználásával.
A panorámakép készítés lépései:
1. Kulcspontok keresése a képeken
2. Összetartozó kulcspontok keresése
3. Az egyező kulcspontok alapján a képek transzformálása
4. A transzformált képek összefűzése

## 1. Kulcspontok keresése

A betöltött képeken az OpenCV függvénykönyvtárban elérhető *ORB detektort*<sup>1</sup> használom a kulcspontok kereséséhez. Az ORB dokumentációja szerint a futtatásakor a SIFT algoritmussal keresi a sarokpontokat és azok leírását.

A képet ehhez előbb szürkeárnyalatossá kell konvertálni a `cvtColor()` fügvénnyel, amely hívásakor meg kell adni a koncertálandó képet illetve azt, hogy milyen konverziót hajtson végre - ez jelen esetben `COLOR_BGR2GRAY`.  
A második lépésben a *jellemződetektor* obijektumot kell létrehozni. A konstruktor az `ORB_create()` paranccsal hívható. Ennek az általam használt paramétere az `nfeatures`. A megadott értékkel beállítható, hogy legfeljebb hány pontot adjon vissza az elárás.
A `.detectAndCompute()` függvény egy lépésben adja vissza a jellemző pontokat (`keyPts`) és a pontok leírását (`descriptors`).

## 2. Kulcspontok párosítása

Az első lépésben, a két képen külön-külön megtalált *jellemző képpontok* párosításához a *BFMatcher* osztályt<sup>2</sup> használom. A brute-force párosító működésének lényege, hogy az egyik kép minden egyes jellemző pontját  megpróbálja megtalálni a másik képen. Ehhez az előző lépésben kiszámolt és a `desctiptors` tömbben tárolt vektorokat használja. Az eljárás az egyezőnek talált pontok távolságát is meghatározza.

Ez a folyamat a pontokat párosító obijektum létrehozásával kezdődik, amihez a `BFMatcher_create()` konstruktort kell meghívni. Mivel a kulcspontokat és leírásukat az ORB detektorral készítettük, ezért a párosító működését a `NORM_HAMMING` paraméterrel kell beállítani.
A pontok párosítását végül a `.knnMatch()` függvény végzi el, amelyben a `k` paraméterrel megadott értékkel lehet maximalizálni, hogy az egyik kép egy-egy kulcspontjához hány kulcspontot párosítson a másik képről.

Az így megtalált kulcspontok közül ki kell választani azokat, amelyek egy bizonyos távolságnál közelebb vannak egymáshoz. A megfelelően közeli kulcspontok az alábbi ciklussal szelektálódnak ki:

```python
bestPts = []
for m, n in commonPts:
    if m.distance < 0.6 * n.distance:
        bestPts.append(m)
```

## 3. A képek transzformálása

Az első lépésben megtalált majd a második lépésben párosított és a távolságuk alapján szelektált *jellemző képpontok* felhasználásával a képeket perspektívikusan torzíthatjuk. A torzítás célja, hogy a két kép egymást átfedő területén a jellegzetes képpontok *ugyanoda* essenek, így a képek pontosan összeilleszthetők lesznek.

Ha elegendő összetartozó képpont van a két képen, akkor meg kell határozni képekhez tartozó *homográfiát*. Az így kapott vektorok a két kép síkjának eltérését mutatják meg. A homográfiák meghatározása a `findHomography()` függvénnyel történik, amelynek a közös pontok leírását és a számításhoz szükséges algoritmus megnevezését (jelen esetben `RANSAC`) is meg kell adni.   
A homográfia birtokában a képek a `perspectiveTransform()` segítségével transzformálhatók.   

# HIVATKOZÁSOK
Egyes kódrészletek forrása: https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html  
<sup>1</sup> https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html  
<sup>2</sup> https://docs.opencv.org/3.4.0/d3/da1/classcv_1_1BFMatcher.html