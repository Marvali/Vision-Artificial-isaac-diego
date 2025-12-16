%% Detectar objetos
% La función detectObjects devuelve los centroides y las cajas delimitadoras 
% de los objetos detectados. También devuelve la máscara binaria, que tiene 
% el mismo tamaño que el frame de entrada. Los píxeles con un valor de 1 
% corresponden al primer plano, y los píxeles con un valor de 0 corresponden 
% al fondo. 
% La función realiza la segmentación del movimiento utilizando el detector 
% de primer plano. A continuación, realiza operaciones morfológicas en la 
% máscara binaria resultante para eliminar los píxeles con ruido y rellenar 
% los huecos de las manchas restantes.
function [mask, detections] = detectObjects(obj, frame, morpho)
        
        % Detecta el primer plano (foreground)
        mask = obj.detector(frame);
        
        % Aplica operaciones morfológicas para eliminar el ruido y rellenar los huecos.
            % mask = imfill(mask, 'holes');
            % if ~isempty(morpho.form)
            %     se = strel(morpho.form, morpho.r);
            %     % mask = imdilate(mask, se); 
            %     mask = imopen(mask, se);
            % end
        % Realiza un análisis de blobs para encontrar los componentes conectados.
        [detections.area, detections.centroids, detections.bboxes] = obj.blobAnalyser(mask);
        
    end