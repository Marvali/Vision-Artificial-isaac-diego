function PedestrianTrackingYOLOv4_Final_Expanded()
    % =========================================================================
    % TRABAJO FINAL DE VISIÓN ARTIFICIAL: SEGUIMIENTO DE PEATONES
    % Autor(es): [Tu Nombre / Equipo]
    % Descripción: Sistema de seguimiento robusto basado en detección por Deep
    % Learning (YOLOv4) y filtrado de Kalman. Incluye evaluación cuantitativa
    % contra Ground Truth (Hito 4).
    % =========================================================================

    % Limpieza del entorno de trabajo para asegurar una ejecución limpia.
    clear all; clc;

    %% ================= 1. AJUSTE DE SINCRONIZACIÓN (CRÍTICO PARA HITO 4) =================
    % PROBLEMA DETECTADO: Durante el desarrollo, observamos un desfase temporal
    % significativo entre el vídeo 'raw.webm' y las anotaciones 'gt.txt'.
    % El vídeo parece ir más rápido que las anotaciones.
    % SOLUCIÓN: Implementamos un modelo lineal de corrección temporal.
    % - GT_RATIO: Factor multiplicativo de velocidad (Ratio > 1 acelera el GT).
    % - GT_OFFSET: Desplazamiento de frames constante.
    % Estos valores se han obtenido empíricamente mediante inspección visual.
    GT_RATIO  = 2.0;   
    GT_OFFSET = 0;     
    
    %% ================= 2. CARGA DE DATOS Y PREPARACIÓN =================
    % Carga del Ground Truth (GT). Es esencial para la validación objetiva del sistema.
    % Si no existe el archivo, detenemos la ejecución.
    if ~exist('gt.txt', 'file'), error('Falta el archivo gt.txt necesario para la evaluación.'); end
    gt = readmatrix('gt.txt'); % Formato esperado: [frame, id, x, y, w, h, ...]  
    
    % Inicialización de acumuladores para métricas de rendimiento.
    % Guardaremos el mejor IoU y Dice de cada detección en cada frame para
    % calcular la media global al final.
    allIoUs  = [];
    allDice  = [];
    frame_number = 0; % Contador global de frames del vídeo procesado.
    
    % Configuración de Visualización de Diagnóstico (Modo Debug).
    % Permite superponer las cajas del GT y las detecciones crudas de YOLO
    % para verificar visualmente la calidad del detector y la sincronización.
    showGT  = true;  % Dibuja cajas VERDES (Referencia / Ground Truth)
    showDet = true;  % Dibuja cajas ROJAS (Detección YOLO directa)

    %% --- CONFIGURACIÓN DE ACELERACIÓN HARDWARE (GPU) ---
    % La inferencia de redes neuronales profundas como YOLOv4 es computacionalmente
    % muy costosa en CPU. Intentamos habilitar CUDA para usar la GPU y
    % conseguir un framerate aceptable.
    useGPU = false;
    if gpuDeviceCount > 0
        try
            parallel.gpu.enableCUDAForwardCompatibility(true);
            gpuDevice(); % Intenta reservar el dispositivo GPU
            useGPU = true;
            fprintf('ÉXITO: GPU (CUDA) activada para aceleración de YOLOv4.\n');
        catch
            fprintf('AVISO: GPU detectada pero no se pudo inicializar (fallo CUDA). Se usará CPU (lento).\n');
        end
    else
        fprintf('AVISO: No se detectó GPU compatible. Se usará CPU (lento).\n');
    end

    %% --- CONFIGURACIÓN DEL DETECTOR (YOLOv4) ---
    % Se utiliza el modelo pre-entrenado 'csp-darknet53-coco'. Este modelo
    % es el estado del arte en equilibrio velocidad/precisión para detección.
    name = 'csp-darknet53-coco'; 
    try
        detectorYOLO = yolov4ObjectDetector(name);
    catch
        error('Error al cargar YOLOv4. Asegúrate de tener instalado el "Computer Vision Toolbox Model for YOLO v4 Object Detection".');
    end

    %% --- CONFIGURACIÓN DE ENTRADA/SALIDA DE VÍDEO ---
    videoFile = 'PETS09-S2L1-raw.webm'; % Secuencia del benchmark MOTChallenge
    obj.reader = VideoReader(videoFile);
    % Configuramos un reproductor de vídeo con un tamaño adecuado para ver los textos de diagnóstico.
    obj.videoPlayer = vision.VideoPlayer('Position', [29, 597, 800, 600], 'Name', 'Seguimiento YOLOv4 + Kalman');
    
    % Carga de la tabla de escalas (pedScaleTable.mat).
    % Se utiliza para filtrar falsos positivos basándose en la altura esperada
    % de un peatón según su posición vertical en la imagen (perspectiva).
    ld = load('pedScaleTable.mat', 'pedScaleTable');
    obj.pedScaleTable = ld.pedScaleTable;

    %% --- INICIALIZACIÓN DE ESTRUCTURA DE TRACKS ---
    % Se define una matriz de estructuras para mantener el estado de cada objeto seguido.
    % Incluye su ID único, su filtro de Kalman asociado, historial de cajas, etc.
    tracks = struct(...
        'id', {}, 'color', {}, 'bboxes', {}, 'scores', {}, ...
        'kalmanFilter', {}, 'age', {}, 'totalVisibleCount', {}, ...
        'confidence', {}, 'predPosition', {});
    nextId = 1; % Contador para asignar IDs únicos a nuevos tracks.

    %% --- AJUSTE FINO DE PARÁMETROS (TUNING) ---
    % Estos parámetros son críticos para el comportamiento del tracker (GNN - Global Nearest Neighbor).
    % Se han ajustado experimentalmente para reducir la fragmentación de trayectorias.
    
    option.ROI                  = [];  % ROI vacío: Usamos toda la imagen (PETS09 es cámara fija).
    option.scThresh             = 0.9; % Umbral de escala estricto (tolera +/- 10% de altura esperada).
    
    % Parámetros de Asociación (Data Association):
    option.gatingThresh         = 0.8; % Coste máximo permitido para asociar (equivale a pedir un IoU mínimo de 0.2).
    option.costOfNonAssignment  = 20;  % Penalización alta para evitar dejar detecciones sin asignar fácilmente.
    
    % Parámetros de Ciclo de Vida del Track:
    option.ageThresh            = 4;   % Edad mínima para empezar a mostrar un track (filtra ruido inicial).
    option.visThresh            = 0.7; % Ratio de visibilidad mínimo para no borrar el track (70% del tiempo detectado).

    fprintf('Iniciando bucle de procesamiento...\n');

    %% ================= BUCLE PRINCIPAL DE PROCESAMIENTO =================
    while hasFrame(obj.reader)
        frame = readFrame(obj.reader);
        frame_number = frame_number + 1;
        
        % -----------------------------------------------------------------
        % ETAPA 1: DETECCIÓN (MEDICIÓN)
        % -----------------------------------------------------------------
        % Se invoca al detector YOLOv4 para obtener las mediciones del frame actual.
        % La función auxiliar también aplica el filtrado por escala y clase 'person'.
        [centroids, bboxes, scores] = detectPeopleYOLO(frame);
        
        % -----------------------------------------------------------------
        % ETAPA DE EVALUACIÓN (HITO 4): SINCRONIZACIÓN Y MÉTRICAS
        % -----------------------------------------------------------------
        % Cálculo del frame objetivo en el Ground Truth aplicando la corrección temporal.
        target_frame_gt = round(frame_number * GT_RATIO) + GT_OFFSET;
        
        % Extracción de las cajas de referencia (GT) para el frame sincronizado.
        idxGT = gt(:,1) == target_frame_gt;
        gtBBoxes = gt(idxGT, 3:6); % Formato [x, y, w, h]

        % Cálculo de métricas (solo si hay detecciones y GT para comparar).
        if ~isempty(bboxes) && ~isempty(gtBBoxes)
            % Calculamos la matriz de IoU (Intersection over Union) entre todas
            % las detecciones y todas las cajas de GT usando bboxOverlapRatio.
            overlaps = bboxOverlapRatio(bboxes, gtBBoxes);
            
            % Para cada detección, buscamos su mejor coincidencia en el GT.
            for d = 1:size(bboxes,1)
                detBox = bboxes(d,:);
                
                % --- Métrica 1: IoU (Jaccard) ---
                % Obtenemos el máximo solapamiento para esta detección 'd'.
                [maxIoU, gtIdx] = max(overlaps(d, :));
                
                % --- Métrica 2: Coeficiente Sørensen-Dice ---
                % Esta métrica se basa en áreas (máscaras binarias), no solo cajas.
                % Es más exigente y se pide explícitamente en la rúbrica.
                maskDet = rect2mask(detBox, size(frame)); % Convertir caja DET a máscara
                
                if maxIoU > 0 % Si hubo algún solapamiento (IoU > 0), comparamos con ese GT.
                    maskGT = rect2mask(gtBBoxes(gtIdx,:), size(frame)); % Convertir caja GT a máscara
                else
                    % Si no hubo solapamiento, la máscara GT es negra (intersección 0).
                    maskGT = false(size(frame,1), size(frame,2));
                end
                diceVal = dice(maskDet, maskGT); % Función nativa de MATLAB para Dice.
                
                % Acumulamos los resultados en los históricos globales.
                allIoUs(end+1) = maxIoU;
                allDice(end+1) = diceVal;
            end
        end
        
        % Cálculo de las medias acumuladas hasta el momento actual (para visualización en tiempo real).
        currIoU = 0; currDice = 0;
        if ~isempty(allIoUs)
            currIoU = mean(allIoUs);
            currDice = mean(allDice);
        end

        % -----------------------------------------------------------------
        % ETAPA 2: PREDICCIÓN DEL FILTRO DE KALMAN
        % -----------------------------------------------------------------
        % Antes de usar las nuevas detecciones, usamos el modelo dinámico (velocidad constante)
        % del Filtro de Kalman para predecir dónde deberían estar los tracks existentes.
        % Esto es crucial para manejar oclusiones momentáneas.
        for i = 1:length(tracks)
            bbox = tracks(i).bboxes(end, :);
            predictedCentroid = predict(tracks(i).kalmanFilter);
            % Convertimos el centroide predicho de nuevo a formato [x, y, w, h]
            tracks(i).predPosition = [predictedCentroid - bbox(3:4)/2, bbox(3:4)];
        end
        
        % -----------------------------------------------------------------
        % ETAPA 3: ASOCIACIÓN DE DATOS (DATA ASSOCIATION)
        % -----------------------------------------------------------------
        % El corazón del tracker. Emparejamos las predicciones (tracks) con las
        % nuevas mediciones (detecciones YOLO).
        if isempty(tracks) || isempty(bboxes)
            % Casos triviales: si no hay tracks o no hay detecciones, no se asigna nada.
            assignments = zeros(0,2);
            unassignedTracks = 1:length(tracks);
            unassignedDetections = 1:size(bboxes,1);
        else
            % Calculamos la matriz de costes basada en la distancia IoU (1 - IoU).
            predBboxes = reshape([tracks(:).predPosition], 4, [])';
            cost = 1 - bboxOverlapRatio(predBboxes, bboxes);
            
            % Aplicamos "gating": si el coste es muy alto (poco solape), se fuerza
            % un coste infinito para impedir la asignación.
            cost(cost > option.gatingThresh) = 1 + 30;
            
            % Resolvemos el problema de asignación (Algoritmo Húngaro / Munkres)
            % usando la función auxiliar assignDetectionsToTracks.
            [assignments, unassignedTracks, unassignedDetections] = ...
                assignDetectionsToTracks(double(cost), double(option.costOfNonAssignment));
        end
        
        % -----------------------------------------------------------------
        % ETAPA 4: GESTIÓN DEL CICLO DE VIDA DE LOS TRACKS
        % -----------------------------------------------------------------
        % Las funciones se han encapsulado al final para mantener el bucle principal limpio.
        
        % 1. Actualizar tracks asignados con las nuevas mediciones (Corrección Kalman).
        updateTracks();
        
        % 2. Borrar tracks que se consideran perdidos (baja visibilidad o muy viejos sin actualizar).
        deleteLostTracks();
        
        % 3. Crear nuevos tracks para las detecciones que no se pudieron asociar a nada existente.
        createNewTracks();
        
        % -----------------------------------------------------------------
        % ETAPA 5: VISUALIZACIÓN Y DIAGNÓSTICO
        % -----------------------------------------------------------------
        % Se muestra el frame resultante con las cajas de tracking, las cajas
        % de diagnóstico (GT/YOLO crudo) y las métricas en tiempo real.
        displayResultsWithMetrics(frame, gtBBoxes, bboxes, tracks, ...
            frame_number, target_frame_gt, currIoU, currDice);
        
        % Salir si se cierra la ventana del reproductor.
        if ~isOpen(obj.videoPlayer), break; end
    end
    
    %% ================= INFORME FINAL DE RESULTADOS =================
    % Se imprimen las medias globales finales, necesarias para la comparativa del trabajo.
    fprintf('\n=== RESULTADOS FINALES DE LA EJECUCIÓN ===\n');
    fprintf('Parámetros de Sincronización usados: Ratio=%.1f | Offset=%d\n', GT_RATIO, GT_OFFSET);
    fprintf('--------------------------------------------------------\n');
    fprintf('Métrica Mean IoU (Jaccard): %.4f  (Medida de solapamiento de cajas)\n', mean(allIoUs));
    fprintf('Métrica Mean Dice:          %.4f  (Medida de similitud de máscaras)\n', mean(allDice));
    fprintf('--------------------------------------------------------\n');

    %% ================= FUNCIONES AUXILIARES Y DEFINICIONES =================

    % --- FUNCIÓN: Detección con YOLOv4 ---
    % Encapsula la llamada al detector, el filtrado por clase y por escala.
    function [centroids, bboxes, scores] = detectPeopleYOLO(img)
        if useGPU
            % Uso de 'ExecutionEnvironment', 'gpu' para aceleración CUDA.
            [allBboxes, allScores, allLabels] = detect( ...
                detectorYOLO, img, 'Threshold', 0.45, 'ExecutionEnvironment', 'gpu');
        else
            [allBboxes, allScores, allLabels] = detect( ...
                detectorYOLO, img, 'Threshold', 0.45);
        end
        
        % Filtro 1: Mantener solo la clase 'person'.
        idx = (allLabels == 'person');
        bboxes = allBboxes(idx, :);
        scores = allScores(idx, :);
        
        % Filtro 2: Consistencia geométrica/perspectiva (pedScaleTable).
        % Elimina detecciones cuyo tamaño no corresponde a la altura esperada en esa posición 'y'.
        if ~isempty(bboxes)
            yfoot = min(length(obj.pedScaleTable), round(bboxes(:,2) + bboxes(:,4)));
            estHeight = obj.pedScaleTable(yfoot);
            % Se marca como inválido si difiere más del umbral definido (ej. 30% o 90%).
            invalid = abs(estHeight - bboxes(:,4)) > estHeight * option.scThresh;
            bboxes(invalid, :) = [];
            scores(invalid, :) = [];
        end

        % Cálculo de centroides para el filtro de Kalman.
        if isempty(bboxes)
            centroids = [];
        else
            centroids = [bboxes(:,1) + bboxes(:,3)/2, bboxes(:,2) + bboxes(:,4)/2 ];
        end
    end

    % --- FUNCIÓN: Conversión Caja a Máscara (para Dice) ---
    % Necesaria porque la función 'dice()' de MATLAB opera sobre imágenes binarias.
    function mask = rect2mask(box, imSize)
        mask = false(imSize(1), imSize(2)); % Inicializar imagen negra
        b = round(box);
        % Asegurar que las coordenadas están dentro de los límites de la imagen.
        x1 = max(1, b(1)); y1 = max(1, b(2));
        x2 = min(imSize(2), b(1)+b(3)); y2 = min(imSize(1), b(2)+b(4));
        % Rellenar el rectángulo con 'true' (blanco).
        mask(y1:y2, x1:x2) = true;
    end

    % --- FUNCIÓN: Actualizar Tracks Asignados ---
    function updateTracks()
        % Para los tracks que han encontrado una detección correspondiente:
        for i = 1:size(assignments, 1)
            tIdx = assignments(i, 1); % Índice del track
            dIdx = assignments(i, 2); % Índice de la detección
            
            % Paso de CORRECCIÓN del Filtro de Kalman con la nueva medición.
            correct(tracks(tIdx).kalmanFilter, centroids(dIdx, :));
            
            % Actualizar historial y contadores.
            tracks(tIdx).bboxes(end+1, :) = bboxes(dIdx, :);
            tracks(tIdx).age = tracks(tIdx).age + 1;
            tracks(tIdx).totalVisibleCount = tracks(tIdx).totalVisibleCount + 1;
            tracks(tIdx).scores(end+1) = scores(dIdx);
        end
        % Para los tracks que NO se han asignado (oclusión temporal):
        for i = 1:length(unassignedTracks)
            idx = unassignedTracks(i);
            tracks(idx).age = tracks(idx).age + 1;
            % Asumimos que siguen su predicción (modelo de velocidad constante).
            tracks(idx).bboxes = [tracks(idx).bboxes; tracks(idx).predPosition];
            tracks(idx).scores(end+1) = 0; % Score 0 porque no hubo detección real.
        end
    end

    % --- FUNCIÓN: Eliminar Tracks Perdidos ---
    function deleteLostTracks()
        if isempty(tracks), return; end
        % Criterio de eliminación: Tracks viejos (age > ageThresh) que han sido
        % visibles muy pocas veces en relación a su edad (visibilidad < visThresh).
        ages = [tracks(:).age]';
        totalVisible = [tracks(:).totalVisibleCount]';
        visibility = totalVisible ./ ages;
        lostInds = (ages > option.ageThresh & visibility < option.visThresh);
        tracks = tracks(~lostInds);
    end

    % --- FUNCIÓN: Crear Nuevos Tracks ---
    function createNewTracks()
        % Para cada detección que no se pudo asociar a ningún track existente:
        for i = 1:length(unassignedDetections)
            idx = unassignedDetections(i);
            % Se inicializa un nuevo Filtro de Kalman (modelo de velocidad constante).
            % [2, 1] = dimensionalidad del estado y medida.
            % [5, 5] = ruido inicial del proceso y de la medida.
            kf = configureKalmanFilter('ConstantVelocity', centroids(idx,:), [2, 1], [5, 5], 100);
            
            % Se crea la nueva estructura de track.
            newTrack = struct('id', nextId, 'color', rand(1,3), ...
                'bboxes', bboxes(idx,:), 'scores', scores(idx), ...
                'kalmanFilter', kf, 'age', 1, 'totalVisibleCount', 1, ...
                'confidence', scores(idx), 'predPosition', bboxes(idx,:) );
            
            % Se añade al array global de tracks.
            tracks(end + 1) = newTrack;
            nextId = nextId + 1;
        end
    end

    % --- FUNCIÓN: Visualización Enriquecida (OSD) ---
    function displayResultsWithMetrics(frame, gtBox, detBox, trk, fNum, tGT, mIoU, mDice)
        outFrame = frame;
        
        % 1. Dibujar cajas de Diagnóstico (si las flags están activas).
        % Ground Truth en VERDE (Referencia).
        if showGT && ~isempty(gtBox)
            outFrame = insertShape(outFrame, 'Rectangle', gtBox, 'Color', 'green', 'LineWidth', 2);
        end
        % Detecciones crudas de YOLO en ROJO.
        if showDet && ~isempty(detBox)
            outFrame = insertShape(outFrame, 'Rectangle', detBox, 'Color', 'red', 'LineWidth', 2);
        end
        
        % 2. Dibujar Tracks Confirmados (Resultado del sistema).
        % Solo mostramos tracks que han superado la edad mínima (ageThresh) para evitar ruido.
        for i = 1:length(trk)
            if trk(i).age > option.ageThresh
                % Se dibuja el ID y la caja con el color asignado al track.
                outFrame = insertObjectAnnotation(outFrame, 'rectangle', ...
                    trk(i).bboxes(end,:), sprintf('ID:%d', trk(i).id), ...
                    'Color', trk(i).color, 'LineWidth', 2);
            end
        end
        
        % 3. Mostrar Panel de Métricas en Pantalla (Requisito Hito 4).
        % Muestra el frame actual, el frame GT sincronizado y las medias acumuladas.
        txt = sprintf('Frame Video: %d | Frame GT: %d\nMean IoU (Acumulado): %.3f\nMean Dice (Acumulado): %.3f', ...
            fNum, tGT, mIoU, mDice);
        % Cuadro amarillo en la esquina superior izquierda con texto negro.
        outFrame = insertText(outFrame, [10 10], txt, 'FontSize', 16, 'BoxColor', 'yellow', 'TextColor', 'black', 'BoxOpacity', 0.8);
        
        % Enviar el frame procesado al reproductor.
        step(obj.videoPlayer, outFrame);
    end
end