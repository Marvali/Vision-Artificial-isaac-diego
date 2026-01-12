function PedestrianTrackingYOLOv4_Final_Expanded()
    % Limpieza inicial
    clear all; clc;

    %% ================= 1. AJUSTE DE SINCRONIZACIÓN (HITO 4) =================
    % Ajusta estos valores observando las cajas verdes vs las personas
    GT_RATIO  = 2.0;   % Velocidad (Prueba 1.0 si va bien, 2.0 si el GT va lento)
    GT_OFFSET = 0;     % Adelanto/Retraso de frames (positivo o negativo)
    
    %% ================= 2. CARGA DE DATOS =================
    % Ground Truth
    if ~exist('gt.txt', 'file'), error('Falta el archivo gt.txt'); end
    gt = readmatrix('gt.txt');   
    
    % Acumuladores de métricas
    allIoUs  = [];
    allDice  = [];
    frame_number = 0;
    
    % Configuración Visual
    showGT  = true;  % Dibuja cajas VERDES (Ground Truth)
    showDet = true;  % Dibuja cajas ROJAS (YOLO)

    %% --- CONFIGURACIÓN GPU ---
    useGPU = false;
    if gpuDeviceCount > 0
        try
            parallel.gpu.enableCUDAForwardCompatibility(true);
            gpuDevice();
            useGPU = true;
            fprintf('GPU activada para YOLOv4\n');
        catch
            fprintf('GPU detectada pero no usable, usando CPU\n');
        end
    else
        fprintf('No se detectó GPU, usando CPU\n');
    end

    %% --- CONFIGURACIÓN DETECTOR ---
    name = 'csp-darknet53-coco'; 
    try
        detectorYOLO = yolov4ObjectDetector(name);
    catch
        error('Instala el modelo YOLOv4 support package.');
    end

    %% --- CONFIGURACIÓN VIDEO ---
    videoFile = 'PETS09-S2L1-raw.webm'; 
    obj.reader = VideoReader(videoFile);
    obj.videoPlayer = vision.VideoPlayer('Position', [29, 597, 800, 600]);
    
    ld = load('pedScaleTable.mat', 'pedScaleTable');
    obj.pedScaleTable = ld.pedScaleTable;

    %% --- INICIALIZACIÓN TRACKS ---
    tracks = struct(...
        'id', {}, 'color', {}, 'bboxes', {}, 'scores', {}, ...
        'kalmanFilter', {}, 'age', {}, 'totalVisibleCount', {}, ...
        'confidence', {}, 'predPosition', {});
    nextId = 1;

    %% --- PARÁMETROS ---
    % IMPORTANTE: ROI vacío para PETS09 (toda la imagen)
    option.ROI                  = []; 
    option.scThresh             = 0.9;   
    option.gatingThresh         = 0.8;   
    option.costOfNonAssignment  = 20;    
    option.timeWindowSize       = 16;    
    option.confidenceThresh     = 0.35;  
    option.ageThresh            = 4;     
    option.visThresh            = 0.7;   

    %% ================= BUCLE PRINCIPAL =================
    while hasFrame(obj.reader)
        frame = readFrame(obj.reader);
        frame_number = frame_number + 1;
        
        % 1. DETECCIÓN YOLOv4
        [centroids, bboxes, scores] = detectPeopleYOLO(frame);
        
        % 2. OBTENER GT SINCRONIZADO (Lógica nueva insertada aquí)
        target_frame_gt = round(frame_number * GT_RATIO) + GT_OFFSET;
        idxGT = gt(:,1) == target_frame_gt;
        gtBBoxes = gt(idxGT, 3:6); % [x, y, w, h]

        % 3. CÁLCULO DE MÉTRICAS (IoU + Dice)
        if ~isempty(bboxes) && ~isempty(gtBBoxes)
            % Matriz de coincidencia
            overlaps = bboxOverlapRatio(bboxes, gtBBoxes);
            
            for d = 1:size(bboxes,1)
                detBox = bboxes(d,:);
                
                % Mejor IoU para esta detección
                [maxIoU, gtIdx] = max(overlaps(d, :));
                
                % Dice (Máscaras)
                maskDet = rect2mask(detBox, size(frame));
                if maxIoU > 0
                    maskGT = rect2mask(gtBBoxes(gtIdx,:), size(frame));
                else
                    maskGT = false(size(frame,1), size(frame,2));
                end
                diceVal = dice(maskDet, maskGT);
                
                % Guardar
                allIoUs(end+1) = maxIoU;
                allDice(end+1) = diceVal;
            end
        end

        % Medias actuales para mostrar en pantalla
        currIoU = 0; currDice = 0;
        if ~isempty(allIoUs)
            currIoU = mean(allIoUs);
            currDice = mean(allDice);
        end

        % 4. PREDICCIÓN KALMAN (Tu código original)
        for i = 1:length(tracks)
            bbox = tracks(i).bboxes(end, :);
            predictedCentroid = predict(tracks(i).kalmanFilter);
            tracks(i).predPosition = ...
                [predictedCentroid - bbox(3:4)/2, bbox(3:4)];
        end
        
        % 5. ASIGNACIÓN (Tu código original)
        if isempty(tracks) || isempty(bboxes)
            assignments = zeros(0,2);
            unassignedTracks = 1:length(tracks);
            unassignedDetections = 1:size(bboxes,1);
        else
            predBboxes = reshape([tracks(:).predPosition], 4, [])';
            cost = 1 - bboxOverlapRatio(predBboxes, bboxes);
            cost(cost > option.gatingThresh) = 1 + 30;
            [assignments, unassignedTracks, unassignedDetections] = ...
                assignDetectionsToTracks(double(cost), ...
                                         double(option.costOfNonAssignment));
        end
        
        % 6. ACTUALIZAR / BORRAR / CREAR (Funciones expandidas abajo)
        updateTracks();
        deleteLostTracks();
        createNewTracks();
        
        % 7. VISUALIZACIÓN ENRIQUECIDA
        displayResultsWithMetrics(frame, gtBBoxes, bboxes, tracks, ...
            frame_number, target_frame_gt, currIoU, currDice);
        
        if ~isOpen(obj.videoPlayer), break; end
    end
    
    fprintf('\n=== RESULTADOS FINALES ===\n');
    fprintf('Ratio: %.1f | Offset: %d\n', GT_RATIO, GT_OFFSET);
    fprintf('Mean IoU: %.4f\n', mean(allIoUs));
    fprintf('Mean Dice: %.4f\n', mean(allDice));


    %% ================= FUNCIONES =================

    function [centroids, bboxes, scores] = detectPeopleYOLO(img)
        if useGPU
            [allBboxes, allScores, allLabels] = detect( ...
                detectorYOLO, img, 'Threshold', 0.45, 'ExecutionEnvironment', 'gpu');
        else
            [allBboxes, allScores, allLabels] = detect( ...
                detectorYOLO, img, 'Threshold', 0.45);
        end
        
        % Filtrar solo 'person'
        idx = (allLabels == 'person');
        bboxes = allBboxes(idx, :);
        scores = allScores(idx, :);
        
        % Filtro por escala
        if ~isempty(bboxes)
            yfoot = min(length(obj.pedScaleTable), round(bboxes(:,2) + bboxes(:,4)));
            estHeight = obj.pedScaleTable(yfoot);
            invalid = abs(estHeight - bboxes(:,4)) > estHeight * option.scThresh;
            bboxes(invalid, :) = [];
            scores(invalid, :) = [];
        end

        if isempty(bboxes)
            centroids = [];
        else
            centroids = [bboxes(:,1) + bboxes(:,3)/2, bboxes(:,2) + bboxes(:,4)/2 ];
        end
    end

    % Función auxiliar para métrica Dice (convertir caja a máscara)
    function mask = rect2mask(box, imSize)
        mask = false(imSize(1), imSize(2));
        b = round(box);
        x1 = max(1, b(1)); y1 = max(1, b(2));
        x2 = min(imSize(2), b(1)+b(3)); y2 = min(imSize(1), b(2)+b(4));
        mask(y1:y2, x1:x2) = true;
    end

    function updateTracks()
        % Asignados
        for i = 1:size(assignments, 1)
            tIdx = assignments(i, 1);
            dIdx = assignments(i, 2);
            correct(tracks(tIdx).kalmanFilter, centroids(dIdx, :));
            tracks(tIdx).bboxes(end+1, :) = bboxes(dIdx, :);
            tracks(tIdx).age = tracks(tIdx).age + 1;
            tracks(tIdx).totalVisibleCount = tracks(tIdx).totalVisibleCount + 1;
            tracks(tIdx).scores(end+1) = scores(dIdx);
        end
        % No asignados
        for i = 1:length(unassignedTracks)
            idx = unassignedTracks(i);
            tracks(idx).age = tracks(idx).age + 1;
            tracks(idx).bboxes = [tracks(idx).bboxes; tracks(idx).predPosition];
            tracks(idx).scores(end+1) = 0;
        end
    end

    function deleteLostTracks()
        if isempty(tracks), return; end
        ages = [tracks(:).age]';
        totalVisible = [tracks(:).totalVisibleCount]';
        visibility = totalVisible ./ ages;
        lostInds = (ages > option.ageThresh & visibility < option.visThresh);
        tracks = tracks(~lostInds);
    end

    function createNewTracks()
        for i = 1:length(unassignedDetections)
            idx = unassignedDetections(i);
            kf = configureKalmanFilter('ConstantVelocity', centroids(idx,:), [2, 1], [5, 5], 100);
            newTrack = struct('id', nextId, 'color', rand(1,3), ...
                'bboxes', bboxes(idx,:), 'scores', scores(idx), ...
                'kalmanFilter', kf, 'age', 1, 'totalVisibleCount', 1, ...
                'confidence', scores(idx), 'predPosition', bboxes(idx,:) );
            tracks(end + 1) = newTrack;
            nextId = nextId + 1;
        end
    end

    function displayResultsWithMetrics(frame, gtBox, detBox, trk, fNum, tGT, mIoU, mDice)
        outFrame = frame;
        
        
        
        % Dibujar cajas GT (Verde) y Detección (Rojo)
        if showGT && ~isempty(gtBox)
            outFrame = insertShape(outFrame, 'Rectangle', gtBox, 'Color', 'green', 'LineWidth', 2);
        end
        if showDet && ~isempty(detBox)
            outFrame = insertShape(outFrame, 'Rectangle', detBox, 'Color', 'red', 'LineWidth', 2);
        end

        % Dibujar Tracks
        for i = 1:length(tracks)
            if tracks(i).age > option.ageThresh
                outFrame = insertObjectAnnotation(outFrame, 'rectangle', ...
                    tracks(i).bboxes(end,:), sprintf('ID:%d', tracks(i).id), ...
                    'Color', tracks(i).color);
            end
        end
        
        % Mostrar Métricas
        txt = sprintf('Frame: %d (GT: %d)\nMean IoU: %.3f\nMean Dice: %.3f', ...
            fNum, tGT, mIoU, mDice);
        outFrame = insertText(outFrame, [10 10], txt, 'FontSize', 18, 'BoxColor', 'yellow', 'TextColor', 'black');
        
        step(obj.videoPlayer, outFrame);
    end
end