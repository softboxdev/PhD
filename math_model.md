# **ПОШАГОВЫЙ ПЛАН ВЫЧИСЛЕНИЙ И АЛГОРИТМЫ С ДИАГРАММАМИ Mermaid**

## **ОБЩАЯ БЛОК-СХЕМА СИСТЕМЫ**

```mermaid
flowchart TD
    A[Начало: Загрузка ТЗ] --> B[Шаг 1: NLP-анализ]
    B --> C[Шаг 2: Классификация паттернов]
    C --> D[Шаг 3: Графовое моделирование]
    D --> E[Шаг 4: Оптимизация конфигурации]
    E --> F[Шаг 5: Симуляция качества]
    F --> G[Шаг 6: Интерпретация и рекомендации]
    G --> H[Конец: Архитектурное решение]
    
    style A fill:#e1f5fe
    style H fill:#e8f5e8
```

---

## **ШАГ 1: NLP-АНАЛИЗ ТЗ - ДЕТАЛЬНАЯ БЛОК-СХЕМА**

```mermaid
flowchart TD
    A1[ТЗ документ PDF/DOCX] --> B1[Предобработка текста]
    
    B1 --> C1{Выбор стратегии парсинга}
    C1 -->|Структурированный| D1[Парсинг секций и заголовков]
    C1 -->|Неструктурированный| E1[Сегментация по семантике]
    
    D1 --> F1[Извлечение таблиц и диаграмм]
    E1 --> G1[Кластеризация тем]
    
    F1 --> H1
    G1 --> H1[Нормализация текста]
    
    H1 --> I1{Применение DACIE алгоритма}
    
    subgraph I1 [DACIE Алгоритм]
        I1_1[Графовое представление документа] --> I1_2[Нейро-символьное<br>извлечение сущностей]
        I1_2 --> I1_3[Онтологическое обогащение]
        I1_3 --> I1_4[Контрастивное обучение<br>для разрешения неоднозначностей]
    end
    
    I1 --> J1[Структурированные требования<br>в формате JSON-LD]
    
    J1 --> K1[Валидация против онтологии АСУ ТП]
    K1 --> L1{Качество извлечения > 0.9?}
    L1 -->|Да| M1[Переход к Шагу 2]
    L1 -->|Нет| N1[Коррекция через<br>активное обучение]
    N1 --> A1
    
    style I1 fill:#f3e5f5
    style L1 fill:#fff3e0
```

### **Алгоритм 1.1: DACIE - Извлечение сущностей**
```python
def dacie_entity_extraction(document_text, ontology):
    """
    Domain-Adaptive Compositional Information Extractor
    """
    # 1. Графовое представление документа
    doc_graph = build_document_graph(document_text)
    
    # 2. Нейронный компонент: кодирование
    neural_embeddings = neural_encoder(doc_graph)
    
    # 3. Символьный компонент: применение правил
    symbolic_features = apply_domain_rules(doc_graph, ontology)
    
    # 4. Композиционное объединение
    for node in doc_graph.nodes:
        # 4.1. Взвешивание источников
        gate = compute_gate(neural_embeddings[node], symbolic_features[node])
        
        # 4.2. Остаточное соединение
        combined = gate * neural_embeddings[node] + \
                  (1 - gate) * symbolic_features[node]
        
        # 4.3. Контрастивное обучение
        if is_ambiguous(node):
            hard_negatives = generate_hard_negatives(node, ontology)
            loss = contrastive_loss(combined, hard_negatives)
            update_weights(loss)
    
    # 5. Извлечение сущностей
    entities = extract_from_combined_representation(doc_graph)
    
    return entities
```

### **Диаграмма последовательности Шага 1**

```mermaid
sequenceDiagram
    participant User as Пользователь
    participant DocParser as Парсер документов
    participant DACIE as DACIE Алгоритм
    participant Ontology as Онтология АСУ ТП
    participant Validator as Валидатор
    participant Step2 as Шаг 2
    
    User->>DocParser: Загрузить ТЗ документ
    DocParser->>DocParser: Предобработка текста
    DocParser->>DocParser: Сегментация на секции
    DocParser->>DACIE: Структурированный текст
    
    DACIE->>DACIE: Построение графа документа
    DACIE->>DACIE: Нейронное кодирование
    DACIE->>Ontology: Запрос доменных правил
    Ontology-->>DACIE: Правила и ограничения
    DACIE->>DACIE: Нейро-символьное извлечение
    DACIE->>DACIE: Контрастивное обучение
    DACIE-->>Validator: Извлеченные сущности
    
    Validator->>Ontology: Проверка соответствия
    Ontology-->>Validator: Результаты проверки
    
    alt Качество > 0.9
        Validator-->>Step2: Структурированные требования
    else Качество <= 0.9
        Validator-->>User: Запрос на разметку
        User-->>Validator: Корректирующая разметка
        Validator->>DACIE: Обновление модели
        DACIE->>DACIE: Активное обучение
        Validator-->>Step2: Исправленные требования
    end
```

---

## **ШАГ 2: КЛАССИФИКАЦИЯ ПАТТЕРНОВ - ДЕТАЛЬНАЯ БЛОК-СХЕМА**

```mermaid
flowchart TD
    A2[Структурированные требования] --> B2[Извлечение признаков]
    
    B2 --> C2{Анализ характеристик проекта}
    
    subgraph C2 [Характеристики]
        C2_1[Real-time требования] --> C2_2[Безопасность SIL/ASIL]
        C2_3[Интеграционная сложность] --> C2_4[Масштабируемость]
        C2_5[Ограничения команды] --> C2_6[Бюджет/сроки]
    end
    
    C2 --> D2[Построение вектора признаков<br>размерности 256]
    
    D2 --> E2[Многоуровневая классификация]
    
    subgraph E2 [Классификационный конвейер]
        E2_1[Дерево решений<br>для быстрой фильтрации] --> E2_2[Ансамбль градиентного бустинга<br>XGBoost/LightGBM]
        E2_2 --> E2_3[Нейронная сеть<br>с вниманием]
    end
    
    E2 --> F2[Ранжирование паттернов]
    
    F2 --> G2{Генерация гибридных паттернов?}
    G2 -->|Да| H2[Комбинирование топ-3 паттернов]
    G2 -->|Нет| I2
    
    H2 --> I2[Оценка комбинаций<br>методом Монте-Карло]
    
    I2 --> J2[Фильтрация по ограничениям]
    J2 --> K2[Топ-5 рекомендуемых паттернов]
    
    K2 --> L2[Переход к Шагу 3]
    
    style E2 fill:#e8f5e8
```

### **Алгоритм 2.1: Многоуровневая классификация паттернов**
```python
def multi_level_pattern_classification(feature_vector, constraints):
    """
    Трехуровневый классификационный конвейер
    """
    patterns = []
    
    # Уровень 1: Быстрая фильтрация (дерево решений)
    candidate_patterns = level1_decision_tree(feature_vector)
    
    # Уровень 2: Точная классификация (градиентный бустинг)
    for pattern in candidate_patterns:
        # 2.1. Извлечение доменно-специфичных признаков
        domain_features = extract_domain_features(pattern, feature_vector)
        
        # 2.2. Ансамблевая классификация
        ensemble_prediction = gradient_boosting_ensemble(domain_features)
        
        # 2.3. Калибровка уверенности
        calibrated_confidence = calibrate_confidence(ensemble_prediction)
        
        if calibrated_confidence > 0.7:
            patterns.append({
                'pattern': pattern,
                'confidence': calibrated_confidence,
                'features': domain_features
            })
    
    # Уровень 3: Нейронная дооценка с вниманием
    ranked_patterns = []
    for item in patterns:
        # 3.1. Внимание к ключевым требованиям
        attention_weights = compute_attention(
            item['features'], 
            constraints['critical_requirements']
        )
        
        # 3.2. Нейронное переранжирование
        final_score = neural_reranking(item, attention_weights)
        
        ranked_patterns.append({
            **item,
            'final_score': final_score,
            'attention_weights': attention_weights
        })
    
    # Сортировка по итоговому score
    ranked_patterns.sort(key=lambda x: x['final_score'], reverse=True)
    
    return ranked_patterns[:5]  # Топ-5 паттернов
```

### **Диаграмма последовательности Шага 2**

```mermaid
sequenceDiagram
    participant Step1 as Шаг 1
    participant FeatureExtractor as Извлекатель признаков
    participant Classifier as Классификатор
    participant PatternDB as База паттернов
    participant HybridGen as Генератор гибридов
    participant Evaluator as Оценщик
    participant Step3 as Шаг 3
    
    Step1->>FeatureExtractor: Структурированные требования
    FeatureExtractor->>FeatureExtractor: Извлечение 256 признаков
    FeatureExtractor->>Classifier: Вектор признаков
    
    Classifier->>Classifier: Уровень 1: Дерево решений
    Classifier->>Classifier: Уровень 2: Градиентный бустинг
    Classifier->>Classifier: Уровень 3: Нейронное внимание
    Classifier->>PatternDB: Запрос похожих паттернов
    PatternDB-->>Classifier: Исторические паттерны
    
    Classifier-->>Evaluator: Классифицированные паттерны
    
    Evaluator->>Evaluator: Оценка соответствия ограничениям
    
    alt Нужны гибридные решения
        Evaluator->>HybridGen: Запрос на генерацию гибридов
        HybridGen->>HybridGen: Комбинирование топ-3 паттернов
        HybridGen->>HybridGen: Оценка комбинаций Монте-Карло
        HybridGen-->>Evaluator: Гибридные паттерны
    end
    
    Evaluator->>Evaluator: Ранжирование всех вариантов
    Evaluator-->>Step3: Топ-5 рекомендуемых паттернов
```

---

## **ШАГ 3: ГРАФОВОЕ МОДЕЛИРОВАНИЕ - ДЕТАЛЬНАЯ БЛОК-СХЕМА**

```mermaid
flowchart TD
    A3[Выбранный паттерн + требования] --> B3[Инициализация графа]
    
    B3 --> C3[Создание вершин-компонентов]
    
    C3 --> D3[Добавление ребер-зависимостей]
    
    D3 --> E3{Применение GNN оптимизации}
    
    subgraph E3 [GNN Optimization Pipeline]
        E3_1[Graph Convolutional Network] --> E3_2[Attention Mechanism]
        E3_2 --> E3_3[Graph Pooling Layer]
        E3_3 --> E3_4[Message Passing]
    end
    
    E3 --> F3[Обучение с подкреплением<br>для оптимизации связей]
    
    F3 --> G3[Валидация графа]
    
    G3 --> H3{Граф соответствует<br>всем ограничениям?}
    
    H3 -->|Да| I3[Аннотация графа метаданными]
    H3 -->|Нет| J3[Коррекция через<br>обратное распространение]
    J3 --> E3
    
    I3 --> K3[Сохранение графовой модели<br>в формате GraphML]
    
    K3 --> L3[Переход к Шагу 4]
    
    style E3 fill:#f3e5f5
```

### **Алгоритм 3.1: GNN-оптимизация графа архитектуры**
```python
def gnn_architecture_optimization(initial_graph, requirements):
    """
    Графовая нейронная сеть для оптимизации архитектуры
    """
    # Инициализация GNN
    gnn = GraphNeuralNetwork(
        in_channels=initial_graph.node_features_dim,
        hidden_channels=128,
        out_channels=64,
        num_layers=3
    )
    
    # Обучение с подкреплением для оптимизации
    for epoch in range(100):
        # 1. Прямое распространение в GNN
        node_embeddings = gnn(initial_graph.x, initial_graph.edge_index)
        
        # 2. Вычисление reward-функции
        rewards = compute_rewards(
            node_embeddings, 
            initial_graph, 
            requirements
        )
        
        # 3. Policy Gradient обновление
        if epoch % 10 == 0:
            # 3.1. Поиск оптимальной конфигурации
            optimal_config = find_optimal_configuration(
                node_embeddings, 
                method='beam_search'
            )
            
            # 3.2. Вычисление преимуществ
            advantages = compute_advantages(rewards, optimal_config)
            
            # 3.3. Обновление политики
            gnn.update_policy(advantages)
        
        # 4. Динамическое изменение графа
        if should_restructure_graph(initial_graph, rewards):
            initial_graph = restructure_graph(
                initial_graph, 
                node_embeddings
            )
    
    # Финальная оптимизация
    optimized_graph = apply_final_optimization(initial_graph)
    
    return optimized_graph
```

### **Диаграмма последовательности Шага 3**

```mermaid
sequenceDiagram
    participant Step2 as Шаг 2
    participant GraphBuilder as Построитель графа
    participant GNN as GNN Оптимизатор
    participant RLAgent as RL Агент
    participant Validator as Валидатор графа
    participant Step4 as Шаг 4
    
    Step2->>GraphBuilder: Паттерн + требования
    GraphBuilder->>GraphBuilder: Создание начального графа
    GraphBuilder->>GNN: Граф для оптимизации
    
    loop 100 эпох обучения
        GNN->>GNN: Graph Convolution
        GNN->>GNN: Attention Mechanism
        GNN->>RLAgent: Node embeddings
        
        RLAgent->>RLAgent: Вычисление reward
        RLAgent->>RLAgent: Policy Gradient update
        
        alt Каждые 10 эпох
            RLAgent->>GNN: Градиенты для обновления
            GNN->>GNN: Обновление весов
        end
        
        alt Нужна реструктуризация
            RLAgent->>GraphBuilder: Запрос на изменение графа
            GraphBuilder->>GraphBuilder: Реструктуризация
            GraphBuilder->>GNN: Обновленный граф
        end
    end
    
    GNN->>Validator: Оптимизированный граф
    
    Validator->>Validator: Проверка ограничений
    
    alt Граф валиден
        Validator->>Validator: Аннотация метаданными
        Validator-->>Step4: GraphML модель
    else Граф невалиден
        Validator-->>GNN: Ошибки валидации
        GNN->>GNN: Дополнительная оптимизация
        GNN->>Validator: Исправленный граф
    end
```

---

## **ШАГ 4: ОПТИМИЗАЦИЯ КОНФИГУРАЦИИ - ДЕТАЛЬНАЯ БЛОК-СХЕМА**

```mermaid
flowchart TD
    A4[Графовая модель системы] --> B4[Генерация пространства решений]
    
    B4 --> C4[Многокритериальная оптимизация]
    
    subgraph C4 [Оптимизационные алгоритмы]
        C4_1[NSGA-II<br>для Pareto фронта] --> C4_2[Симулированный отжиг]
        C4_2 --> C4_3[Муравьиный алгоритм]
    end
    
    C4 --> D4[Балансировка нагрузок]
    
    D4 --> E4[Добавление резервирования]
    
    E4 --> F4{Валидация отказоустойчивости}
    
    F4 -->|Прошла| G4[Генерация конфигурационных файлов]
    F4 -->|Не прошла| H4[Усиление резервирования]
    H4 --> E4
    
    subgraph G4 [Форматы вывода]
        G4_1[Docker Compose] --> G4_2[Kubernetes Manifests]
        G4_2 --> G4_3[Terraform скрипты]
        G4_3 --> G4_4[Ansible playbooks]
    end
    
    G4 --> I4[Оценка стоимости и ресурсов]
    
    I4 --> J4[Переход к Шагу 5]
    
    style C4 fill:#e8f5e8
```

### **Алгоритм 4.1: Многокритериальная оптимизация NSGA-II**
```python
def nsga2_architecture_optimization(solution_space, objectives, constraints):
    """
    NSGA-II для многокритериальной оптимизации архитектуры
    """
    # Инициализация популяции
    population = initialize_population(solution_space, size=100)
    
    for generation in range(50):
        # 1. Оценка популяции
        fitness_values = []
        for individual in population:
            # Вычисление всех целевых функций
            fitness = {
                'performance': evaluate_performance(individual),
                'cost': evaluate_cost(individual),
                'reliability': evaluate_reliability(individual),
                'maintainability': evaluate_maintainability(individual)
            }
            
            # Проверка ограничений
            if check_constraints(individual, constraints):
                fitness_values.append(fitness)
            else:
                # Штраф за нарушение ограничений
                fitness_values.append(apply_penalty(fitness))
        
        # 2. Недоминируемая сортировка
        fronts = non_dominated_sort(fitness_values)
        
        # 3. Вычисление crowding distance
        for front in fronts:
            crowding_distance(front, fitness_values)
        
        # 4. Селекция
        selected = selection(population, fronts, size=len(population)//2)
        
        # 5. Кроссовер и мутация
        offspring = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i+1]
            
            # Кроссовер архитектур
            child1, child2 = architectural_crossover(parent1, parent2)
            
            # Мутация
            if random.random() < 0.1:
                child1 = mutate_architecture(child1)
            if random.random() < 0.1:
                child2 = mutate_architecture(child2)
            
            offspring.extend([child1, child2])
        
        # 6. Формирование новой популяции
        population = selected + offspring
    
    # Выбор лучших решений из Pareto фронта
    pareto_front = get_pareto_front(population, fitness_values)
    
    return pareto_front[:3]  # Топ-3 решения
```

### **Диаграмма последовательности Шага 4**

```mermaid
sequenceDiagram
    participant Step3 as Шаг 3
    participant SolutionSpace as Пространство решений
    participant NSGA2 as NSGA-II Оптимизатор
    participant LoadBalancer as Балансировщик нагрузки
    participant Redundancy as Менеджер резервирования
    participant ConfigGen as Генератор конфигураций
    participant CostEstimator as Оценщик стоимости
    participant Step5 as Шаг 5
    
    Step3->>SolutionSpace: Графовая модель
    SolutionSpace->>SolutionSpace: Генерация вариантов размещения
    SolutionSpace->>NSGA2: Пространство решений
    
    loop 50 поколений
        NSGA2->>NSGA2: Оценка целевых функций
        NSGA2->>NSGA2: Недоминируемая сортировка
        NSGA2->>NSGA2: Вычисление crowding distance
        NSGA2->>NSGA2: Селекция
        NSGA2->>NSGA2: Кроссовер и мутация
    end
    
    NSGA2->>LoadBalancer: Оптимизированные конфигурации
    
    LoadBalancer->>LoadBalancer: Балансировка CPU/памяти/сети
    LoadBalancer->>Redundancy: Сбалансированные конфигурации
    
    Redundancy->>Redundancy: Добавление резервирования
    Redundancy->>Redundancy: Валидация отказоустойчивости
    
    Redundancy->>ConfigGen: Конфигурации с резервированием
    
    ConfigGen->>ConfigGen: Генерация Docker/K8s файлов
    ConfigGen->>ConfigGen: Генерация инфраструктурных скриптов
    
    ConfigGen->>CostEstimator: Конфигурационные файлы
    
    CostEstimator->>CostEstimator: Расчет стоимости
    CostEstimator->>CostEstimator: Оценка ресурсов
    
    CostEstimator-->>Step5: Оптимизированные конфигурации + стоимость
```

---

## **ШАГ 5: СИМУЛЯЦИЯ КАЧЕСТВА - ДЕТАЛЬНАЯ БЛОК-СХЕМА**

```mermaid
flowchart TD
    A5[Оптимизированные конфигурации] --> B5[Создание симуляционной модели]
    
    B5 --> C5{Выбор симуляционного движка}
    
    C5 -->|Дискретная событийная| D5[SimPy или собственный движок]
    C5 -->|Агентное моделирование| E5[Mesa или Repast]
    C5 -->|Аналитическая модель| F5[Очереди Массового обслуживания]
    
    D5 --> G5
    E5 --> G5
    F5 --> G5[Настройка параметров симуляции]
    
    G5 --> H5[Запуск сценариев]
    
    subgraph H5 [Типовые сценарии]
        H5_1[Нормальная работа] --> H5_2[Пиковая нагрузка]
        H5_2 --> H5_3[Частичные отказы]
        H5_3 --> H5_4[Полный отказ компонента]
    end
    
    H5 --> I5[Сбор метрик в реальном времени]
    
    I5 --> J5[Статистический анализ]
    
    J5 --> K5[Выявление узких мест]
    
    K5 --> L5{Найдены критические проблемы?}
    
    L5 -->|Да| M5[Генерация рекомендаций по оптимизации]
    L5 -->|Нет| N5[Подготовка отчета о качестве]
    
    M5 --> N5
    
    N5 --> O5[Переход к Шагу 6]
    
    style H5 fill:#fff3e0
```

### **Алгоритм 5.1: Дискретно-событийная симуляция**
```python
def discrete_event_simulation(configuration, scenarios):
    """
    Дискретно-событийная симуляция архитектуры АСУ ТП
    """
    class ASU_TP_Simulation:
        def __init__(self, config):
            self.env = simpy.Environment()
            self.config = config
            self.metrics = defaultdict(list)
            
            # Инициализация компонентов
            self.init_components()
            
            # Инициализация мониторов
            self.init_monitors()
        
        def init_components(self):
            """Создание симуляционных моделей компонентов"""
            self.components = {}
            for comp in self.config['components']:
                if comp['type'] == 'sensor':
                    self.components[comp['id']] = SensorModel(
                        env=self.env, 
                        config=comp
                    )
                elif comp['type'] == 'controller':
                    self.components[comp['id']] = ControllerModel(
                        env=self.env, 
                        config=comp
                    )
                # ... другие типы компонентов
        
        def run_scenario(self, scenario, duration):
            """Запуск сценария симуляции"""
            # Запуск процессов
            for process in scenario['processes']:
                self.env.process(self.execute_process(process))
            
            # Запуск мониторинга
            self.env.process(self.monitor_metrics())
            
            # Запуск сбоев (если есть)
            if 'failures' in scenario:
                for failure in scenario['failures']:
                    self.env.process(self.inject_failure(failure))
            
            # Запуск симуляции
            self.env.run(until=duration)
            
            return self.collect_results()
        
        def execute_process(self, process):
            """Выполнение бизнес-процесса"""
            while True:
                # Генерация событий согласно процессу
                yield self.env.timeout(process['interval'])
                
                # Обработка события
                start_time = self.env.now
                
                # Прохождение через компоненты
                for step in process['steps']:
                    component = self.components[step['component']]
                    yield component.process(step['data'])
                
                # Запись метрик
                processing_time = self.env.now - start_time
                self.metrics['processing_times'].append(processing_time)
    
    # Запуск симуляции для всех сценариев
    results = {}
    for scenario_name, scenario_config in scenarios.items():
        sim = ASU_TP_Simulation(configuration)
        scenario_results = sim.run_scenario(
            scenario_config, 
            duration=scenario_config['duration']
        )
        results[scenario_name] = scenario_results
    
    return results
```

### **Диаграмма последовательности Шага 5**

```mermaid
sequenceDiagram
    participant Step4 as Шаг 4
    participant SimBuilder as Построитель симуляции
    participant SimEngine as Симуляционный движок
    participant MetricCollector as Сборщик метрик
    participant Analyzer as Анализатор
    participant BottleneckDetector as Детектор узких мест
    participant Step6 as Шаг 6
    
    Step4->>SimBuilder: Конфигурации системы
    SimBuilder->>SimBuilder: Создание моделей компонентов
    SimBuilder->>SimEngine: Симуляционная модель
    
    par Для каждого сценария
        SimEngine->>SimEngine: Инициализация сценария
        SimEngine->>SimEngine: Запуск процессов
        SimEngine->>SimEngine: Инжекция событий/сбоев
        
        SimEngine->>MetricCollector: События в реальном времени
        MetricCollector->>MetricCollector: Агрегация метрик
    end
    
    MetricCollector->>Analyzer: Собранные метрики
    
    Analyzer->>Analyzer: Статистический анализ
    Analyzer->>BottleneckDetector: Распределения метрик
    
    BottleneckDetector->>BottleneckDetector: Анализ производительности
    BottleneckDetector->>BottleneckDetector: Выявление узких мест
    
    alt Найдены проблемы
        BottleneckDetector->>BottleneckDetector: Генерация рекомендаций
        BottleneckDetector-->>Step6: Отчет + рекомендации
    else Проблем не найдено
        BottleneckDetector-->>Step6: Отчет о качестве
    end
```

---

## **ШАГ 6: ИНТЕРПРЕТАЦИЯ И РЕКОМЕНДАЦИИ - ДЕТАЛЬНАЯ БЛОК-СХЕМА**

```mermaid
flowchart TD
    A6[Результаты всех этапов] --> B6[Агрегация данных]
    
    B6 --> C6[Генерация объяснений XAI]
    
    subgraph C6 [Explainable AI методы]
        C6_1[SHAP анализ важности] --> C6_2[LIME локальные объяснения]
        C6_2 --> C6_3[Attention визуализация]
        C6_3 --> C6_4[Контрастные примеры]
    end
    
    C6 --> D6[Формирование рекомендаций]
    
    D6 --> E6[Приоритизация по бизнес-ценности]
    
    E6 --> F6[Создание интерактивного отчета]
    
    subgraph F6 [Компоненты отчета]
        F6_1[Сводная панель] --> F6_2[Детализация решений]
        F6_2 --> F6_3[Альтернативы и trade-offs]
        F6_3 --> F6_4[План реализации]
    end
    
    F6 --> G6[Валидация с экспертом]
    
    G6 --> H6{Эксперт подтверждает решение?}
    
    H6 -->|Да| I6[Финальное архитектурное решение]
    H6 -->|Нет| J6[Итеративная корректировка]
    J6 --> K6{На каком этапе проблема?}
    
    K6 -->|Требования| L6[Коррекция на Шаге 1]
    K6 -->|Паттерны| M6[Коррекция на Шаге 2]
    K6 -->|Архитектура| N6[Коррекция на Шаге 3]
    K6 -->|Конфигурация| O6[Коррекция на Шаге 4]
    
    I6 --> P6[Экспорт всех артефактов]
    
    P6 --> Q6[Конец: Готовое решение]
    
    style C6 fill:#f3e5f5
```

### **Алгоритм 6.1: Генерация объяснимых рекомендаций**
```python
def generate_explainable_recommendations(all_steps_data, simulation_results):
    """
    Генерация объяснимых рекомендаций с использованием XAI
    """
    recommendations = {
        'summary': {},
        'detailed_recommendations': [],
        'explanations': {},
        'alternatives': [],
        'implementation_plan': {}
    }
    
    # 1. SHAP анализ для важности решений
    shap_values = compute_shap_values(all_steps_data)
    recommendations['explanations']['feature_importance'] = shap_values
    
    # 2. Генерация контрастных примеров
    contrastive_examples = generate_contrastive_examples(
        all_steps_data['final_decision'],
        all_steps_data['alternative_decisions']
    )
    recommendations['explanations']['contrastive'] = contrastive_examples
    
    # 3. Локальные объяснения LIME
    for key_decision in all_steps_data['key_decisions']:
        lime_explanation = lime_explain(
            decision=key_decision,
            data=all_steps_data,
            num_features=10
        )
        recommendations['explanations']['local'][key_decision['id']] = \
            lime_explanation
    
    # 4. Формирование рекомендаций на основе объяснений
    recommendations['detailed_recommendations'] = formulate_recommendations(
        all_steps_data, 
        recommendations['explanations']
    )
    
    # 5. Приоритизация по бизнес-ценности
    prioritized_recommendations = prioritize_by_business_value(
        recommendations['detailed_recommendations'],
        business_objectives=all_steps_data['business_objectives']
    )
    
    # 6. Создание плана реализации
    implementation_plan = create_implementation_plan(
        prioritized_recommendations,
        team_capabilities=all_steps_data['team_info']
    )
    
    recommendations['implementation_plan'] = implementation_plan
    
    return recommendations
```

### **Диаграмма последовательности Шага 6**

```mermaid
sequenceDiagram
    participant Steps1_5 as Шаги 1–5
    participant Aggregator as Агрегатор данных
    participant XAI as XAI Модуль
    participant Recommender as Рекомендательная система
    participant Prioritizer as Приоритизатор
    participant ReportGen as Генератор отчётов
    participant Expert as Эксперт‑архитектор
    participant Export as Экспорт модуль

    Steps1_5->>Aggregator: Результаты всех этапов (Шаги 1–5)

    Aggregator->>Aggregator: Объединение данных
    Aggregator->>XAI: Передача агрегированных данных

    XAI->>XAI: SHAP‑анализ важности признаков
    XAI->>XAI: LIME‑анализ локальных объяснений
    XAI->>XAI: Генерация контрастных примеров
    XAI->>Recommender: Передача объяснений и данных

    Recommender->>Recommender: Формирование рекомендаций по оптимизации
    Recommender->>Prioritizer: Передача детальных рекомендаций

    Prioritizer->>Prioritizer: Оценка бизнес‑ценности и рисков
    Prioritizer->>ReportGen: Передача приоритизированных рекомендаций

    ReportGen->>ReportGen: Создание интерактивного отчёта (дашборд + документация)
    ReportGen->>Expert: Отправка отчёта на валидацию

    Expert->>Expert: Анализ рекомендаций и проверка соответствия требованиям

    alt Эксперт согласен с рекомендациями
        Expert->>Export: Подтверждение решения и запуск экспорта
        Export->>Export: Генерация всех артефактов (схемы, код, документация)
        Export-->>Expert: Предоставление готового архитектурного решения
    else Эксперт не согласен с рекомендациями
        Expert->>Expert: Идентификация проблемы и её типа

        alt Проблема в исходных требованиях
            Expert-->>Steps1_5: Запрос на корректировку Шага 1 (требования)
        else Проблема в выбранных паттернах проектирования
            Expert-->>Steps1_5: Запрос на корректировку Шага 2 (паттерны)
        else Проблема в архитектуре системы
            Expert-->>Steps1_5: Запрос на корректировку Шага 3 (архитектура)
        else Проблема в конфигурации окружения
            Expert-->>Steps1_5: Запрос на корректировку Шага 4 (конфигурация)
        end
    end

```

---

## **ПОЛНАЯ ИНТЕГРАЦИОННАЯ ДИАГРАММА**

```mermaid
flowchart TD
    Start[Начало проекта] --> Upload[Загрузка ТЗ]
    Upload --> Step1
    Step1 --> Step2
    Step2 --> Step3
    Step3 --> Step4
    Step4 --> Step5
    Step5 --> Step6
    Step6 --> Decision{Архитектор<br>принимает решение?}
    
    Decision -->|Да| Final[Архитектурное решение готово]
    Decision -->|Нет, корректировка| Feedback[Определение этапа для корректировки]
    
    Feedback -->|Требования| Step1
    Feedback -->|Паттерны| Step2
    Feedback -->|Архитектура| Step3
    Feedback -->|Конфигурация| Step4
    Feedback -->|Качество| Step5
    
    Final --> Export[Экспорт артефактов]
    Export --> Artifacts[Артефакты проекта]
    
    subgraph Step1 [Шаг 1: NLP-анализ]
        S1A[Парсинг ТЗ] --> S1B[DACIE алгоритм] --> S1C[Структурированные требования]
    end
    
    subgraph Step2 [Шаг 2: Классификация паттернов]
        S2A[Извлечение признаков] --> S2B[Многоуровневая классификация] --> S2C[Рекомендованные паттерны]
    end
    
    subgraph Step3 [Шаг 3: Графовое моделирование]
        S3A[Инициализация графа] --> S3B[GNN оптимизация] --> S3C[Валидированный граф]
    end
    
    subgraph Step4 [Шаг 4: Оптимизация конфигурации]
        S4A[Пространство решений] --> S4B[NSGA-II оптимизация] --> S4C[Оптимальная конфигурация]
    end
    
    subgraph Step5 [Шаг 5: Симуляция качества]
        S5A[Создание модели] --> S5B[Запуск сценариев] --> S5C[Анализ метрик]
    end
    
    subgraph Step6 [Шаг 6: Интерпретация]
        S6A[Агрегация данных] --> S6B[XAI объяснения] --> S6C[Рекомендации]
    end
    
    subgraph Artifacts [Выходные артефакты]
        A1[Спецификация требований]
        A2[Архитектурные диаграммы]
        A3[Конфигурационные файлы]
        A4[Отчет о качестве]
        A5[План реализации]
    end
    
    style Start fill:#e1f5fe
    style Final fill:#e8f5e8
    style Artifacts fill:#fff3e0
```

