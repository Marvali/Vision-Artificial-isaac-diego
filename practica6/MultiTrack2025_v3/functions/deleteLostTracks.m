%% Borrar los tracks perdidos
% La función deleteLostTracks elimina los tracks  que han sido invisibles 
% durante demasiados frames consecutivos. También elimina los tracks creadas 
% recientemente que han sido invisibles durante demasiados frames en total. 
    function [tracks ] = deleteLostTracks(tracks, invisibleForTooLong, ageThreshold, visibilityThreshold)
    
    
        if isempty(tracks)
            return;
        end
       
        % Calcula la fracción de la edad del track durante la cual fue visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        
        % Encuentra los índices de los tracks "perdidos".
        lostInds = (ages < ageThreshold & visibility < visibilityThreshold) | [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
                  
        % Borra los tracks "perdidos". 
        % (sólo guarda los que no están perdidos)
        tracks = tracks(~lostInds);
    end