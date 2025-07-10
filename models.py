

@app.post("/train")
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Inicia el entrenamiento distribuido de modelos"""
    def train_task():
        try:
            trainer = get_trainer()
            logger.info(f"Iniciando entrenamiento:")
            
            results = trainer.train_models_distributed(
                task_type=request.task_type,
                selected_models=request.selected_models,
                test_size=request.test_size
            )
            
            if results:
                # Guardar modelos en el directorio correcto
                #trainer.save_models(MODELS_DIR)
                # Guardar resultados
                results_file = os.path.join(TRAINING_RESULTS_DIR, "train_results.json")
                #trainer.save_results(results_file)

                logger.info(f"resultados:: aaaa {results}")
                logger.info(f"✅ Entrenamiento completado. Modelos: {len(results)}")
                filtered_results = dict()

                for key, value in results.items():
                    entry = { x:y for x,y in value.items() if x != 'model'}
                    filtered_results[key] = entry
                return filtered_results
            else:
                logger.warning("⚠️ No se obtuvieron resultados del entrenamiento")
                
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}", exc_info=True)

    a =train_task()
    return {
        "message": "Entrenamiento completado",
        "results": a
    }