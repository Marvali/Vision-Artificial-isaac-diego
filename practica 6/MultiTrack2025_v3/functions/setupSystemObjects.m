%% Crear objetos del sistema
%
% Crea utilidades para leer el vídeo, detectar los objetos en movimiento, y mostrar los resultados.
function obj = setupSystemObjects(videofilename, MinimumBlobArea)
        % Inicializa las I/O de vídeo, Crea objetos para leer un vídeo desde un archivo, 
        % dibujar los objetos rastreados en cada frame y reproducir el vídeo.
        
        % Crea el video reader.
        obj.reader = VideoReader(videofilename);
        
        % Crea los video players,
        % para ver el video
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);
        % para ver la mascara del primer plano.
        obj.maskPlayer = vision.VideoPlayer('Position', [720, 400, 700, 400]);
        
        
        % Crear objetos del sistema para la detección de primer plano 
        % y el análisis de blobs
        
        % El foreground detector se utiliza para segmentar los objetos en 
        % movimiento del fondo. Produce una máscara binaria, en la que el 
        % valor de 1 del píxel corresponde al primer plano y el valor 
        % de 0 corresponde al fondo. 
        
        obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...
            'NumTrainingFrames', 5, 'MinimumBackgroundRatio', 0.7);
        % Crear utilidades para detectar objetos en movimiento usando el modelo de mezcla de gaussianas (GMM) con la técnica Background Subtraction.
        

        % Es probable que los grupos de píxeles conectados en primer plano 
        % se correspondan con objetos en movimiento.  El objeto del sistema 
        % de análisis de blobs se utiliza para encontrar estos grupos 
        % (denominados 'blobs' or 'connected components') y calcular sus 
        % características, como el área, el centroide y el cuadro delimitador (bounding box).
        if ~isempty(MinimumBlobArea)
            obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
                'AreaOutputPort', true, 'CentroidOutputPort', true, ...
                'MinimumBlobArea', MinimumBlobArea);
        end
  end