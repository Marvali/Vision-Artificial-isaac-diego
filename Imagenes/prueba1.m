[bazoI,map]=imread('bazo.bmp');
figure,subplot(1,2,1),imshow(bazoI)
title('Imagenoriginalbazo')

subplot(1,2,2),imhist(bazoI)
title('Histogramabazo')
grisesBazo=ind2gray(bazoI,map);
figure,subplot(1,2,1),imshow(grisesBazo)
title('Histogramaescaladegrises')
eqBazo=histeq(grisesBazo,32);
subplot(1,2,1),imshowclose a(eqBazo)
subplot(1,2,2),imhist(eqBazo)
title('Histogramaecualizado')
