%% Asignar detecciones a tracks
% La asignación de las detecciones de objetos en el frame actual a los tracks
% existentes se realiza minimizando el coste. El coste se define como la 
% probabilidad logarítmica negativa de una detección correspondiente a un track. 
% El algoritmo consta de dos pasos: 
% 
% Paso 1: Calcular el coste de asignar cada detección a cada track utilizando 
% el método de distancia del sistema vision.KalmanFilter object. El coste 
% tiene en cuenta la distancia euclidiana entre el centroide predicho del 
% track y el centroide de la detección. También incluye la confianza de la 
% predicción, que es mantenida por el filtro de Kalman. Los resultados se 
% almacenan en una matriz MxN, donde M es el número de tracks y N el número 
% de detecciones. 
% 
% Paso 2: Resolver el problema de asignación representado por la matriz de 
% costes mediante la función assignDetectionsToTracks. La función toma la 
% matriz de costes y el coste de no asignar ninguna detección a una pista. 
% El valor del coste de no asignar una detección a una pista depende del 
% rango de valores devueltos por el método de la distancia de vision.KalmanFilter. 
% Este valor debe ser ajustado experimentalmente. Si se establece demasiado bajo, 
% aumenta la probabilidad de crear una nueva pista, y puede dar lugar a la 
% fragmentación de la pista. Si se establece demasiado alto, puede dar lugar 
% a una única pista que corresponda a una serie de objetos en movimiento separados. 
% La función assignDetectionsToTracks utiliza la versión de Munkres del 
% algoritmo húngaro para calcular una asignación que minimice el coste total. 
% Devuelve una matriz M x 2 que contiene los índices correspondientes de los 
% tracks y detecciones asignadas en sus dos columnas. También devuelve los 
% índices de  los tracks y detecciones que quedaron sin asignar. 

function [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment(tracks, centroids, costOfNonAssignment)
    
    nTracks = length(tracks);
    nDetections = size(centroids, 1);
    
    % Calcula el coste de asignar cada detección a cada track.
    cost = zeros(nTracks, nDetections);
    for i = 1:nTracks
        % función de coste: Distancias entre las mediciones actuales y
        % las previstas del filtro de Kalman
        cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
    end
    
    % Resuelve el problema de la asignación.
    [assignments, unassignedTracks, unassignedDetections] = assignDetectionsToTracks(cost, costOfNonAssignment);
    
end