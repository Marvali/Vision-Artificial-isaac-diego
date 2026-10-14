function plotTrackID(frame, trackssum, ID, display)

     frame_aux = frame;

     frame_aux = plotBboxes(frame_aux, trackssum, ID, display);
     frame_aux = plotCentroids(frame_aux, trackssum, ID, display);

     fig = figure;
     imshow(frame_aux);
     title(['Tracks ', int2str(ID)]);
     hold on;
    plotTracks(trackssum, ID, display);
    hold off;
end