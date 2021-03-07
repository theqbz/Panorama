# Panorama
###### Panorámakép-készítő program - házi dolgozat Gépi látás (GKLB_INTM038) tárgyból.<br/>Készíti: Borsodi Zoltán (B7PK8Z), mérnökinformatika, Széchenyi István Egyetem

A program Python programozási nyelven készült az OpenCV függvénykönyvtár felhasználásával. Célja, hogy két fotóból panorámaképet készítsen.

## Mi az a panorámakép?

A panorámakép, több egymást átfedő, azonos forgatású képből előállított egyetlen – immár széles – kép. A több kép egyesítésével történő panorámakép alkotást összefűzésnek nevezzük. Az összefűzéshez olyan fotókból kell kiindulni, amelyek azonos állásúak és egymást valamilyen mértékben átfedik. A végeredmény egy egyesített kép lesz, ezeknek az átfedésben lévő fotóknak a felhasználásával.
A panorámakép készítés lépései:
- Kulcspontok keresése a különálló képeken
- Két kép közötti átfedés meghatározása az egyező kulcspontok keresésével
- Az egyező kulcspontok alapján a képek transzformálása
- A transzformált képek összefűzése
