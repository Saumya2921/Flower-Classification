  
        // Flower classes that our model can classify
        const FLOWER_CLASSES = [
            'Daisy', 'Rose', 'Tulip', 'Sunflower', 'Dandelion',
            'Iris', 'Lily', 'Orchid', 'Poppy', 'Lavender',
            'Marigold', 'Carnation', 'Chrysanthemum', 'Peony', 'Hibiscus'
        ];

        let model = null;
        const modelStatus = document.getElementById('modelStatus');
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const imagePreview = document.getElementById('imagePreview');
        const results = document.getElementById('results');

        // Initialize the application
        async function initializeApp() {
            try {
                modelStatus.textContent = 'Loading AI Model...';
                modelStatus.className = 'status loading';
                
                // Create a simple CNN model for demonstration
                // In a real application, you'd load a pre-trained model
                model = await createFlowerClassificationModel();
                
                modelStatus.textContent = 'AI Model Ready!';
                modelStatus.className = 'status ready';
            } catch (error) {
                console.error('Model loading failed:', error);
                modelStatus.textContent = 'Model Loading Failed';
                modelStatus.className = 'status error';
            }
        }

        // Create a simple CNN model (placeholder - in production, load pre-trained weights)
        async function createFlowerClassificationModel() {
            const model = tf.sequential({
                layers: [
                    tf.layers.conv2d({
                        inputShape: [224, 224, 3],
                        filters: 32,
                        kernelSize: 3,
                        activation: 'relu',
                    }),
                    tf.layers.maxPooling2d({ poolSize: 2 }),
                    tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }),
                    tf.layers.maxPooling2d({ poolSize: 2 }),
                    tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }),
                    tf.layers.flatten(),
                    tf.layers.dense({ units: 64, activation: 'relu' }),
                    tf.layers.dropout({ rate: 0.5 }),
                    tf.layers.dense({ units: FLOWER_CLASSES.length, activation: 'softmax' })
                ]
            });

            // Compile the model
            model.compile({
                optimizer: 'adam',
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });

            // Initialize with random weights (in production, load trained weights)
            await model.predict(tf.randomNormal([1, 224, 224, 3])).data();
            
            return model;
        }

        // Handle file upload
        fileInput.addEventListener('change', handleFileSelect);
        uploadArea.addEventListener('drop', handleFileDrop);
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file && file.type.startsWith('image/')) {
                processImage(file);
            }
        }

        function handleFileDrop(event) {
            event.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = event.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                processImage(file);
            }
        }

        function handleDragOver(event) {
            event.preventDefault();
            uploadArea.classList.add('dragover');
        }

        function handleDragLeave(event) {
            event.preventDefault();
            uploadArea.classList.remove('dragover');
        }

        async function processImage(file) {
            // Show image preview
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);

            // Show loading state
            results.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Analyzing flower image...</p>
                </div>
            `;

            try {
                // Create image element for processing
                const img = new Image();
                img.onload = async function() {
                    const predictions = await classifyImage(img);
                    displayResults(predictions);
                };
                img.src = URL.createObjectURL(file);
            } catch (error) {
                console.error('Classification error:', error);
                results.innerHTML = `
                    <p style="color: #dc3545; text-align: center; padding: 20px;">
                        Error processing image. Please try again.
                    </p>
                `;
            }
        }

        async function classifyImage(imgElement) {
            if (!model) {
                throw new Error('Model not loaded');
            }

            // Preprocess image
            const tensor = tf.browser.fromPixels(imgElement)
                .resizeNearestNeighbor([224, 224])
                .toFloat()
                .div(tf.scalar(255.0))
                .expandDims();

            // Get predictions
            const prediction = await model.predict(tensor).data();
            tensor.dispose();

            // Convert to readable format
            const predictions = Array.from(prediction).map((confidence, index) => ({
                className: FLOWER_CLASSES[index],
                confidence: confidence
            }));

            // Sort by confidence
            predictions.sort((a, b) => b.confidence - a.confidence);
            
            return predictions.slice(0, 5); // Top 5 predictions
        }

        function displayResults(predictions) {
            const resultsHTML = predictions.map((pred, index) => {
                const percentage = (pred.confidence * 100).toFixed(1);
                const isTop = index === 0;
                
                return `
                    <div class="prediction-item ${isTop ? 'top' : ''}">
                        <div>
                            <div class="flower-name">${pred.className}</div>
                            <div class="confidence">${percentage}% confidence</div>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${percentage}%"></div>
                        </div>
                    </div>
                `;
            }).join('');

            results.innerHTML = resultsHTML;
        }

        // Initialize app when page loads
        window.addEventListener('load', initializeApp);
    