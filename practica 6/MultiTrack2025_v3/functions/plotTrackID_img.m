function plotTrackID_img(ImagePath, trackssum, ID)

     % lee el frame inicial
     f_i = trackssum(ID).frame_i;
     frameName = ImagePath + sprintf("%06d",f_i) + '.png';
     frame_aux =  imread(frameName);

     % calcula el frame final
     bboxes = cat(1,  trackssum(ID).detections.bbox);
     f_f = f_i + length(bboxes);

     for i = f_i+1 : f_f
         frameName = ImagePath + sprintf("%06d",i) + '.png';
         frame_i =  imread(frameName);
          % se superponen todos los frames desde el inicial al final
         frame_aux = max(frame_i, frame_aux);   
     end


     frame_aux = plotBboxes(frame_aux, trackssum, ID);
    
     fig = figure;
     imshow(frame_aux);
     title([sprintf('Track %d', ID)]);
     plotTracks(trackssum, ID)

end