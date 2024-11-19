//RECIPE
/*
Ecco una ricetta da seguire per implementare un programma multithread in C, che calcola il pagerank di un grafo:

    1. Preparazione della funzione capolettore:
        ◦ Apri il file di input e leggi la prima riga ignorando i commenti (righe che iniziano con '%').
        ◦ Leggi il numero di righe, colonne e archi dalla prima riga significativa. Se la matrice non è quadrata, genera un errore.
        ◦ Inizializza la struttura del grafo con N nodi e crea liste di adiacenza vuote.
        ◦ Crea un HashSet per tenere traccia degli archi già inseriti, evitando duplicati.
        ◦ Per ogni riga successiva del file, leggi una coppia di nodi src e dest, decrementa di 1 (per adattare all'indicizzazione zero) e, se l'arco non esiste già nel grafo, aggiungilo al buffer condiviso.
    2. Sincronizzazione tra capolettore e thread ausiliari:
        ◦ Quando il capolettore legge un arco valido, lo aggiunge al buffer condiviso tramite la funzione produce(pool, arco).
        ◦ Alla fine del file, invia un segnale di terminazione a ciascun thread ausiliario aggiungendo archi con src e dest impostati a -1.
    3. Thread ausiliari per la costruzione del grafo:
        ◦ Ogni thread esegue la funzione thread_function, che consuma archi dal buffer condiviso.
        ◦ Se l'arco consumato è valido, il thread acquisisce il mutex del grafo e aggiunge l'arco alla lista di adiacenza del nodo sorgente.
        ◦ Dopo aver processato un arco, il thread invia un segnale di condizione per risvegliare altri thread in attesa.
    4. Chiusura e terminazione:
        ◦ Quando tutti gli archi sono stati processati e ogni thread riceve il segnale di terminazione (arco.src == -1), il thread termina il proprio lavoro e si chiude.
        ◦ Una volta che tutti i thread ausiliari hanno terminato, il programma può continuare con l'elaborazione del grafo completato.


5. Calcolo del pagerank:
utilizzando il grafo costruito dai thread consumatori viene calcolato il pagerank su di esso in parallelo.
5.1 Inizializzazione :
    • Allocazione delle risorse: Vengono allocati i vettori ranks, old_ranks, e le variabili ausiliarie come sum_dead_ends (che contiene la somma del PageRank dei nodi senza archi uscenti, chiamati "dead end").
    • Valore iniziale di PageRank: Ogni nodo parte con lo stesso valore di PageRank iniziale, che è 1.0 / num_nodes.
    • Suddivisione del lavoro: Il numero totale di nodi viene diviso tra i thread, in modo che ciascun thread calcoli il PageRank di un sottoinsieme di nodi. Gli intervalli sono definiti dai valori start e end per ogni thread.
    • Creazione dei thread: Per ogni thread, viene passato un set di dati tramite la struttura PagerankThreadData, che contiene riferimenti a variabili condivise come i vettori ranks, old_ranks, sum_dead_ends, e il damping factor.
5.2 Calcolo del PageRank:
    • Sincronizzazione iniziale: Ogni thread aspetta che tutti siano pronti usando una barriera (pthread_barrier_wait), così da iniziare l'iterazione simultaneamente.
    • Iterazioni del PageRank:
        ◦ Ogni thread processa i nodi assegnati nell'intervallo start - end.
        ◦ Per ciascun nodo, il thread somma i contributi dei nodi entranti, cioè i nodi che hanno un arco diretto verso il nodo corrente. La somma dei contributi viene calcolata dividendo il PageRank del nodo entrante per il numero di archi uscenti da quel nodo.
        ◦ Il nuovo valore del PageRank del nodo viene calcolato con la formula:
          new_rank=(N1−d​)+d(rank_sum+Nsum_dead_ends​)
          dove d è il damping factor, rank_sum è la somma dei contributi dei nodi entranti e sum_dead_ends è la somma del PageRank dei "dead ends".
        ◦ Il valore di local_delta viene aggiornato per ogni nodo, rappresentando la differenza tra il nuovo e il vecchio valore di PageRank.
    • Sincronizzazione tra iterazioni: Dopo aver calcolato il PageRank per tutti i nodi dell'intervallo, i thread si sincronizzano nuovamente usando una barriera. Il thread principale (thread_id == 0) calcola quindi:
        ◦ Delta globale: Somma i valori di local_delta di tutti i thread per ottenere il valore di delta, che rappresenta la variazione totale tra le iterazioni.
        ◦ Condizione di arresto: Se delta è inferiore alla tolleranza epsilon o il numero massimo di iterazioni è stato raggiunto, imposta stop_flag per terminare l'esecuzione.
        ◦ Aggiornamento dei valori: Copia i nuovi valori di PageRank da ranks a old_ranks e ricalcola sum_dead_ends per la prossima iterazione.
    • Ripetizione: Il ciclo continua finché non viene raggiunta la convergenza (cioè delta è inferiore a epsilon) o il numero massimo di iterazioni.
5.3 Finalizzazione (nella funzione pagerank_parallel):
    • Dopo che tutti i thread hanno completato il loro lavoro (sincronizzati tramite pthread_join), il vettore ranks viene normalizzato affinché la somma dei valori di PageRank sia uguale a 1.
    • Liberazione delle risorse: Viene deallocata la memoria e vengono distrutte le strutture di sincronizzazione come le barriere e i mutex.
5.4 Parallelizzazione dinamica:
    • Ogni thread lavora in parallelo sui nodi assegnati, ma la suddivisione del lavoro è dinamica. I thread non aspettano che tutti gli altri completino il loro lavoro prima di proseguire. Usano una barriera per sincronizzare le iterazioni, ma dentro ogni iterazione calcolano in modo indipendente.
    • Aggiornamento dei nodi dead-end: Il calcolo dei contributi dai nodi "dead end" (nodi senza archi uscenti) viene eseguito una volta per iterazione dal thread principale, garantendo che ogni thread possa utilizzarli durante il calcolo.

6. Stampa del risultato:
Dopo che è finito il calcolo del pagerank devono essere stampati i primi k nodi in ordine di pagerank piu alto. Il parametro k viene passato per parametro.

Puoi scrivere un codice in C che segue questa ricetta?*/

//output 1
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

#define MAX_THREADS 8
#define DAMPING_FACTOR 0.85
#define EPSILON 0.0001
#define MAX_ITER 100

// Struttura per rappresentare il grafo
typedef struct {
    int num_nodes;
    int *out_degree;
    int **incoming_edges; // Matrice di liste di adiacenza
    int *num_incoming;    // Numero di nodi entranti per ciascun nodo
} Graph;

// Dati per i thread del PageRank
typedef struct {
    int start;
    int end;
    Graph *graph;
    double *ranks;
    double *old_ranks;
    double *sum_dead_ends;
    pthread_barrier_t *barrier;
    double *global_delta;
    pthread_mutex_t *delta_mutex;
    int *stop_flag;
} PagerankThreadData;

// Funzione per leggere il grafo dal file (Sezione 1)
Graph *read_graph_from_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    int num_nodes, num_edges;
    fscanf(file, "%d %d", &num_nodes, &num_edges);

    // Inizializzazione del grafo
    Graph *graph = (Graph *)malloc(sizeof(Graph));
    graph->num_nodes = num_nodes;
    graph->out_degree = (int *)calloc(num_nodes, sizeof(int));
    graph->incoming_edges = (int **)malloc(num_nodes * sizeof(int *));
    graph->num_incoming = (int *)calloc(num_nodes, sizeof(int));

    // Lettura degli archi
    for (int i = 0; i < num_edges; i++) {
        int src, dest;
        fscanf(file, "%d %d", &src, &dest);
        src--; // Adatta all'indicizzazione zero
        dest--;
        
        // Aggiungi arco alla lista di adiacenza entrante del nodo dest
        graph->incoming_edges[dest] = realloc(graph->incoming_edges[dest], 
                        (graph->num_incoming[dest] + 1) * sizeof(int));
        graph->incoming_edges[dest][graph->num_incoming[dest]++] = src;
        graph->out_degree[src]++;
    }

    fclose(file);
    return graph;
}

// Funzione per calcolare il PageRank in parallelo (Sezione 4)
void *pagerank_thread(void *arg) {
    PagerankThreadData *data = (PagerankThreadData *)arg;

    int num_nodes = data->graph->num_nodes;
    double *ranks = data->ranks;
    double *old_ranks = data->old_ranks;
    double *sum_dead_ends = data->sum_dead_ends;
    pthread_barrier_t *barrier = data->barrier;
    double *global_delta = data->global_delta;
    pthread_mutex_t *delta_mutex = data->delta_mutex;
    int *stop_flag = data->stop_flag;

    for (int iter = 0; iter < MAX_ITER && !*stop_flag; iter++) {
        double local_delta = 0.0;

        // Calcolo PageRank per ciascun nodo assegnato
        for (int node = data->start; node < data->end; node++) {
            double rank_sum = 0.0;
            
            // Somma dei contributi dei nodi entranti
            for (int i = 0; i < data->graph->num_incoming[node]; i++) {
                int src = data->graph->incoming_edges[node][i];
                rank_sum += old_ranks[src] / data->graph->out_degree[src];
            }

            // Aggiorna il rank con il damping factor
            double new_rank = (1.0 - DAMPING_FACTOR) / num_nodes + 
                              DAMPING_FACTOR * (rank_sum + *sum_dead_ends / num_nodes);
            local_delta += fabs(new_rank - old_ranks[node]);
            ranks[node] = new_rank;
        }

        pthread_barrier_wait(barrier);

        // Aggiornamento del delta globale nel thread 0
        if (data->start == 0) {
            *global_delta = 0.0;
            for (int t = 0; t < MAX_THREADS; t++) {
                *global_delta += local_delta;
            }

            if (*global_delta < EPSILON) {
                *stop_flag = 1;
            }

            // Copia ranks in old_ranks per la prossima iterazione
            for (int i = 0; i < num_nodes; i++) {
                old_ranks[i] = ranks[i];
            }
        }

        pthread_barrier_wait(barrier);
    }
    return NULL;
}

// Funzione principale per gestire i thread del PageRank
void pagerank_parallel(Graph *graph) {
    int num_nodes = graph->num_nodes;

    double *ranks = (double *)malloc(num_nodes * sizeof(double));
    double *old_ranks = (double *)malloc(num_nodes * sizeof(double));
    double sum_dead_ends = 0.0;
    double global_delta = 0.0;
    int stop_flag = 0;

    // Inizializza ranks e old_ranks
    for (int i = 0; i < num_nodes; i++) {
        ranks[i] = 1.0 / num_nodes;
        old_ranks[i] = ranks[i];
        if (graph->out_degree[i] == 0) {
            sum_dead_ends += ranks[i];
        }
    }

    pthread_t threads[MAX_THREADS];
    PagerankThreadData thread_data[MAX_THREADS];
    pthread_barrier_t barrier;
    pthread_mutex_t delta_mutex = PTHREAD_MUTEX_INITIALIZER;

    pthread_barrier_init(&barrier, NULL, MAX_THREADS);

    // Divisione dei nodi tra i thread
    int nodes_per_thread = num_nodes / MAX_THREADS;

    for (int i = 0; i < MAX_THREADS; i++) {
        int start = i * nodes_per_thread;
        int end = (i == MAX_THREADS - 1) ? num_nodes : start + nodes_per_thread;

        thread_data[i] = (PagerankThreadData){
            .start = start, .end = end, .graph = graph, 
            .ranks = ranks, .old_ranks = old_ranks, 
            .sum_dead_ends = &sum_dead_ends,
            .barrier = &barrier, .global_delta = &global_delta,
            .delta_mutex = &delta_mutex, .stop_flag = &stop_flag
        };

        pthread_create(&threads[i], NULL, pagerank_thread, &thread_data[i]);
    }

    // Attesa della fine dei thread
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&barrier);
    pthread_mutex_destroy(&delta_mutex);

    // Stampa i primi k nodi
    int k = 5; // Può essere passato come parametro
    printf("Top %d nodes by PageRank:\n", k);
    for (int i = 0; i < k; i++) {
        printf("Node %d: %f\n", i + 1, ranks[i]);
    }

    free(ranks);
    free(old_ranks);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <graph_file>\n", argv[0]);
        return 1;
    }

    Graph *graph = read_graph_from_file(argv[1]);

    // Calcola il PageRank
    pagerank_parallel(graph);

    // Libera la memoria del grafo
    for (int i = 0; i < graph->num_nodes; i++) {
        free(graph->incoming_edges[i]);
    }
    free(graph->incoming_edges);
    free(graph->out_degree);
    free(graph->num_incoming);
    free(graph);

    return 0;
}

//output 2

typedef struct {
    int N;         // Numero di nodi
    int *out;      // Array che memorizza il numero di archi uscenti per ogni nodo
    inmap *in;     // Array di insiemi di nodi entranti (per ogni nodo)
} grafo;

double *pagerank(grafo *g, double d, double eps, int maxiter, int *numiter) {
    int N = g->N;
    double *x = malloc(N * sizeof(double));  // Vettore per X(t)
    double *xnext = malloc(N * sizeof(double));  // Vettore per X(t+1)
    double *Y = malloc(N * sizeof(double));  // Vettore ausiliario
    double St = 0.0;  // Somma dei dead-end
    int t = 0;

    // Inizializzazione del vettore X(1)
    for (int i = 0; i < N; i++) {
        x[i] = 1.0 / N;
    }

    // Iterazione fino alla convergenza o al massimo numero di iterazioni
    while (t < maxiter) {
        St = 0.0;

        // Calcolo del contributo dei dead-end
        for (int i = 0; i < N; i++) {
            if (g->out[i] == 0) {
                St += x[i];
            }
        }

        // Calcolo di X(t+1) usando la formula data
        for (int j = 0; j < N; j++) {
            double sum_in = 0.0;
            for (int i = 0; i < g->in[j].size; i++) {  // Ciclo sugli archi entranti
                int in_node = g->in[j].nodes[i];
                sum_in += x[in_node] / g->out[in_node];
            }
            xnext[j] = (1 - d) / N + d * sum_in + (d / N) * St;
        }

        // Controllo della convergenza
        double error = 0.0;
        for (int i = 0; i < N; i++) {
            error += fabs(xnext[i] - x[i]);
        }
        if (error < eps) {
            break;
        }

        // Scambio dei vettori x e xnext
        double *temp = x;
        x = xnext;
        xnext = temp;

        t++;
    }

    *numiter = t;
    free(xnext);
    free(Y);
    return x;
}

void *calculate_pagerank_thread(void *arg) {
    // Struttura dei dati del thread
    thread_data *data = (thread_data *)arg;
    
    // Ogni thread calcola una porzione di X(t+1)
    for (int j = data->start; j < data->end; j++) {
        // Calcolo della componente X(t+1)_j
        double sum_in = 0.0;
        for (int i = 0; i < data->g->in[j].size; i++) {
            int in_node = data->g->in[j].nodes[i];
            sum_in += data->x[in_node] / data->g->out[in_node];
        }
        data->xnext[j] = (1 - data->d) / data->N + data->d * sum_in + (data->d / data->N) * data->St;
    }
    
    pthread_exit(NULL);
}

typedef struct {
    grafo *g;
    double *x;      // Vettore X(t)
    double *xnext;  // Vettore X(t+1)
    double d;
    double St;      // Somma dei dead-end
    int N;          // Numero di nodi
    int start;      // Nodo di partenza per il thread
    int end;        // Nodo finale per il thread
} thread_data;

void create_threads(grafo *g, double *x, double *xnext, double d, double St, int N, int num_threads) {
    pthread_t threads[num_threads];
    thread_data tdata[num_threads];

    int chunk_size = N / num_threads;
    for (int i = 0; i < num_threads; i++) {
        tdata[i].g = g;
        tdata[i].x = x;
        tdata[i].xnext = xnext;
        tdata[i].d = d;
        tdata[i].St = St;
        tdata[i].N = N;
        tdata[i].start = i * chunk_size;
        tdata[i].end = (i == num_threads - 1) ? N : (i + 1) * chunk_size;

        // Crea i thread
        pthread_create(&threads[i], NULL, calculate_pagerank_thread, (void *)&tdata[i]);
    }

    // Attendi che tutti i thread abbiano terminato
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

void read_graph(char *filename, grafo *g) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    int r, c, num_edges;
    fscanf(file, "%d %d %d", &r, &c, &num_edges);

    g->N = r;  // Numero di nodi
    g->out = calloc(g->N, sizeof(int));
    g->in = malloc(g->N * sizeof(inmap));

    for (int i = 0; i < num_edges; i++) {
        int u, v;
        fscanf(file, "%d %d", &u, &v);
        u--; v--;  // Modifica per partire da 0

        if (u != v) {
            // Aggiungi arco (u, v)
            add_edge(g, u, v);  // Funzione per aggiornare le strutture dati
        }
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    char *filename = argv[argc - 1];  // Nome del file di input
    double d = 0.85, eps = 1e-6;
    int max_iter = 100, K = 3;

    grafo g;
    read_graph(filename, &g);

    printf("Number of nodes: %d\n", g.N);
    printf("Number of dead-end nodes: %d\n", count_deadends(&g));  // Funzione per contare i dead-end

    int num_iter;
    double *ranks = pagerank(&g, d, eps, max_iter, &num_iter);

    printf("Converged after %d iterations\n", num_iter);
    printf("Sum of ranks: %f (should be 1)\n", sum_ranks(ranks, g.N));  // Somma dei rank
    print_top_k(ranks, g.N, K);  // Funzione per stampare i top K nodi

    free_graph(&g);  // Libera la memoria allocata per il grafo
    free(ranks);
    return 0;
}


//ARCHITECTURAL + RECIPE

/*
Sto scrivendo un programma multithread in C che calcola il pagerank di un grafo. 
Il programma include le seguenti funzionalità:
-lettura della matrice di input
-creazione del grafo utilizzando gli archi letti dalla matrice
-calcolo del pagerank sul grafo costruito
-stampa dei primi nodi col valore di pagerank più alto

Puoi consigliarmi 5 architetture che possano implementare questo progetto?*/

//output 1
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

#define NUM_THREADS 4

typedef struct {
    int N;       // Numero di nodi
    int *out;    // Array del numero di archi uscenti da ogni nodo
    int **in;    // Lista di adiacenza degli archi entranti
} grafo;

// Struttura per passare i dati ai thread
typedef struct {
    grafo *g;
    double *pagerank;
    double *pagerank_next;
    double damping_factor;
    double epsilon;
    int max_iter;
    int thread_id;
    int num_threads;
} thread_data;

pthread_barrier_t barrier;

// Funzione che legge il grafo dal file e lo inizializza
grafo *init_grafo(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Errore apertura file");
        exit(EXIT_FAILURE);
    }

    int N, num_arcs;
    fscanf(file, "%d %d %d", &N, &N, &num_arcs);

    grafo *g = malloc(sizeof(grafo));
    g->N = N;
    g->out = calloc(N, sizeof(int));
    g->in = malloc(N * sizeof(int *));

    for (int i = 0; i < N; i++) {
        g->in[i] = NULL;
    }

    int i, j;
    while (fscanf(file, "%d %d", &i, &j) == 2) {
        if (i != j) {
            g->out[i]++;
            g->in[j] = realloc(g->in[j], (g->out[j] + 1) * sizeof(int));
            g->in[j][g->out[j] - 1] = i;
        }
    }

    fclose(file);
    return g;
}

// Funzione eseguita dai thread per calcolare il PageRank
void *calculate_pagerank(void *arg) {
    thread_data *data = (thread_data *)arg;
    grafo *g = data->g;
    double *pagerank = data->pagerank;
    double *pagerank_next = data->pagerank_next;
    double d = data->damping_factor;
    double eps = data->epsilon;
    int max_iter = data->max_iter;
    int num_threads = data->num_threads;
    int N = g->N;
    
    double init_rank = 1.0 / N;
    for (int i = data->thread_id; i < N; i += num_threads) {
        pagerank[i] = init_rank;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        double diff = 0.0;
        
        for (int i = data->thread_id; i < N; i += num_threads) {
            pagerank_next[i] = (1 - d) / N;
            for (int *in_nodes = g->in[i]; in_nodes != NULL; in_nodes++) {
                int incoming = *in_nodes;
                pagerank_next[i] += d * pagerank[incoming] / g->out[incoming];
            }
        }

        // Sincronizzazione tra i thread
        pthread_barrier_wait(&barrier);

        for (int i = data->thread_id; i < N; i += num_threads) {
            diff += fabs(pagerank_next[i] - pagerank[i]);
            pagerank[i] = pagerank_next[i];
        }

        // Sincronizzazione e verifica della convergenza
        pthread_barrier_wait(&barrier);

        if (diff < eps) {
            break;
        }
    }

    return NULL;
}

// Funzione che stampa i nodi con il valore di PageRank più alto
void print_top_k_nodes(grafo *g, double *pagerank, int k) {
    int *top_nodes = malloc(k * sizeof(int));
    double *top_ranks = malloc(k * sizeof(double));

    for (int i = 0; i < k; i++) {
        top_nodes[i] = i;
        top_ranks[i] = pagerank[i];
    }

    for (int i = k; i < g->N; i++) {
        for (int j = 0; j < k; j++) {
            if (pagerank[i] > top_ranks[j]) {
                top_ranks[j] = pagerank[i];
                top_nodes[j] = i;
                break;
            }
        }
    }

    for (int i = 0; i < k; i++) {
        printf("Nodo %d: PageRank = %f\n", top_nodes[i], top_ranks[i]);
    }

    free(top_nodes);
    free(top_ranks);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_file = argv[1];
    grafo *g = init_grafo(input_file);

    double damping_factor = 0.85;
    double epsilon = 1e-6;
    int max_iter = 100;
    int num_threads = NUM_THREADS;
    int k = 3;  // Numero di nodi da stampare

    double *pagerank = calloc(g->N, sizeof(double));
    double *pagerank_next = calloc(g->N, sizeof(double));

    pthread_t threads[NUM_THREADS];
    thread_data data[NUM_THREADS];

    pthread_barrier_init(&barrier, NULL, NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; i++) {
        data[i].g = g;
        data[i].pagerank = pagerank;
        data[i].pagerank_next = pagerank_next;
        data[i].damping_factor = damping_factor;
        data[i].epsilon = epsilon;
        data[i].max_iter = max_iter;
        data[i].thread_id = i;
        data[i].num_threads = NUM_THREADS;
        pthread_create(&threads[i], NULL, calculate_pagerank, (void *)&data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    print_top_k_nodes(g, pagerank, k);

    pthread_barrier_destroy(&barrier);
    free(pagerank);
    free(pagerank_next);
    free(g->out);
    free(g->in);
    free(g);

    return EXIT_SUCCESS;
}

//output 2

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

#define DAMPING_FACTOR 0.85
#define MAX_ITERATIONS 100
#define TOLERANCE 1e-6

typedef struct {
    int N;           // Numero di nodi del grafo
    int *out_degree; // Grado uscente di ogni nodo
    int **in_list;   // Lista degli archi entranti per ogni nodo
} Graph;

typedef struct {
    int *tasks;         // Array di task (un task è un nodo)
    int task_count;     // Numero di task nel pool
    pthread_mutex_t lock;
} TaskPool;

typedef struct {
    TaskPool *task_pool;
    Graph *graph;
    double *pagerank;
    double *new_pagerank;
    double damping_factor;
    double teleportation;
} ThreadArgs;

// Funzione per rubare un task dal pool
int steal_task(TaskPool *pool) {
    int task = -1;
    pthread_mutex_lock(&pool->lock);
    if (pool->task_count > 0) {
        task = pool->tasks[--pool->task_count];
    }
    pthread_mutex_unlock(&pool->lock);
    return task;
}

// Funzione che calcola il PageRank per un nodo
void calculate_pagerank(Graph *g, double *pagerank, double *new_pagerank, int node, double damping_factor, double teleportation) {
    double sum = 0.0;
    for (int i = 0; i < g->in_list[node][0]; i++) {
        int neighbor = g->in_list[node][i+1]; // Otteniamo il nodo entrante
        sum += pagerank[neighbor] / g->out_degree[neighbor];
    }
    new_pagerank[node] = teleportation + damping_factor * sum;
}

// Funzione eseguita dai thread
void* thread_work(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;
    int task;

    while ((task = steal_task(args->task_pool)) != -1) {
        calculate_pagerank(args->graph, args->pagerank, args->new_pagerank, task, args->damping_factor, args->teleportation);
    }
    return NULL;
}

// Inizializza il pool di task
TaskPool* init_task_pool(int num_tasks) {
    TaskPool *pool = malloc(sizeof(TaskPool));
    pool->tasks = malloc(num_tasks * sizeof(int));
    pool->task_count = num_tasks;
    pthread_mutex_init(&pool->lock, NULL);

    // Aggiungi tutti i nodi come task
    for (int i = 0; i < num_tasks; i++) {
        pool->tasks[i] = i;
    }
    return pool;
}

// Inizializza il grafo
Graph* init_graph(int N) {
    Graph *g = malloc(sizeof(Graph));
    g->N = N;
    g->out_degree = calloc(N, sizeof(int));
    g->in_list = malloc(N * sizeof(int*));
    return g;
}

// Itera il calcolo del PageRank
void run_pagerank(Graph *g, int num_threads) {
    pthread_t threads[num_threads];
    TaskPool *pool = init_task_pool(g->N);
    double *pagerank = calloc(g->N, sizeof(double));
    double *new_pagerank = calloc(g->N, sizeof(double));
    ThreadArgs args = {pool, g, pagerank, new_pagerank, DAMPING_FACTOR, (1.0 - DAMPING_FACTOR) / g->N};

    // Inizializza il pagerank iniziale
    for (int i = 0; i < g->N; i++) {
        pagerank[i] = 1.0 / g->N;
    }

    // Crea i thread
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, thread_work, (void*)&args);
    }

    // Attendi il termine dei thread
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Stampa il risultato del pagerank
    for (int i = 0; i < g->N; i++) {
        printf("Node %d: PageRank = %.6f\n", i, new_pagerank[i]);
    }

    free(pagerank);
    free(new_pagerank);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s num_threads\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    int N = 5;  // Per esempio, numero di nodi del grafo

    Graph *g = init_graph(N);

    // Inserisci i dati del grafo
    // Esempio: inserisci i gradi uscenti e le liste di archi entranti qui

    run_pagerank(g, num_threads);

    return 0;
}


//FLIPPED INTERACTION

/*
implementa il seguente progetto. Fammi tante domande quante te ne servono per restituirmi un codice come lo scriverebbe uno studente universitario di informatica. Fammi una domanda per volta.*/

//output 1
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

typedef struct {
    int N; // Numero di nodi del grafo
    int *out; // Numero di archi uscenti per ogni nodo
    int **in; // Liste di adiacenza per gli archi entranti
} grafo;

// Funzione per creare il grafo
grafo* crea_grafo(int N) {
    grafo *g = (grafo*)malloc(sizeof(grafo));
    g->N = N;
    g->out = (int*)calloc(N, sizeof(int));
    g->in = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) {
        g->in[i] = NULL;
    }
    return g;
}

// Aggiunge un arco al grafo
void aggiungi_arco(grafo *g, int origine, int destinazione) {
    g->out[origine]++;
    // Aggiunge l'origine alla lista di adiacenza del nodo destinazione
    int *new_list = realloc(g->in[destinazione], (g->out[destinazione] + 1) * sizeof(int));
    g->in[destinazione] = new_list;
    g->in[destinazione][g->out[destinazione] - 1] = origine;
}

#define BUFFER_SIZE 1000 // Dimensione base dinamica del buffer

typedef struct {
    int origine, destinazione;
} arco;

arco *buffer;
int buffer_count = 0;
int buffer_size = BUFFER_SIZE;

sem_t empty, full, mutex; // Semafori per sincronizzazione

// Funzione produttore
void* produttore(void* args) {
    FILE *file = (FILE*)args;
    int origine, destinazione;

    while (fscanf(file, "%d %d", &origine, &destinazione) == 2) {
        sem_wait(&empty);
        sem_wait(&mutex);

        // Aggiunge l'arco al buffer
        buffer[buffer_count].origine = origine;
        buffer[buffer_count].destinazione = destinazione;
        buffer_count++;

        sem_post(&mutex);
        sem_post(&full);
    }

    return NULL;
}

// Funzione consumatore
void* consumatore(void* args) {
    grafo *g = (grafo*)args;

    while (1) {
        sem_wait(&full);
        sem_wait(&mutex);

        if (buffer_count == 0) { // Se il buffer è vuoto, termina
            sem_post(&mutex);
            sem_post(&empty);
            return NULL;
        }

        // Estrae l'arco dal buffer
        arco a = buffer[--buffer_count];

        sem_post(&mutex);
        sem_post(&empty);

        // Gestisce l'arco nel grafo
        if (a.origine != a.destinazione) { // Ignora archi autoloop
            aggiungi_arco(g, a.origine, a.destinazione);
        }
    }
}

#define MAX_ITER 100
#define DAMPING_FACTOR 0.85
#define TOLLERANZA 1e-6

typedef struct {
    grafo *g;
    double *pagerank;
    int start, end;
} thread_data;

// Funzione di calcolo del PageRank
void* calcola_pagerank(void *args) {
    thread_data *data = (thread_data*)args;
    grafo *g = data->g;
    double *pagerank = data->pagerank;
    int start = data->start;
    int end = data->end;

    // Implementazione iterativa del calcolo di PageRank
    for (int i = start; i < end; i++) {
        double somma = 0.0;

        for (int j = 0; j < g->out[i]; j++) {
            int in_node = g->in[i][j];
            somma += pagerank[in_node] / g->out[in_node];
        }

        pagerank[i] = (1.0 - DAMPING_FACTOR) / g->N + DAMPING_FACTOR * somma;
    }

    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return -1;
    }

    // Apertura del file
    FILE *file = fopen(argv[1], "r");
    if (!file) {
        perror("Errore apertura file");
        return -1;
    }

    int N = 0, num_arcs = 0;
    fscanf(file, "%d %d %d", &N, &N, &num_arcs); // Legge il numero di nodi e archi

    grafo *g = crea_grafo(N);

    // Inizializzazione buffer e semafori
    buffer = malloc(BUFFER_SIZE * sizeof(arco));
    sem_init(&empty, 0, BUFFER_SIZE);
    sem_init(&full, 0, 0);
    sem_init(&mutex, 0, 1);

    // Creazione dei thread produttore e consumatore
    pthread_t prod, cons[4]; // Un produttore e 4 consumatori
    pthread_create(&prod, NULL, produttore, file);

    for (int i = 0; i < 4; i++) {
        pthread_create(&cons[i], NULL, consumatore, g);
    }

    pthread_join(prod, NULL); // Attende il termine della lettura del file

    for (int i = 0; i < 4; i++) {
        pthread_join(cons[i], NULL); // Attende che i consumatori finiscano
    }

    fclose(file);

    // Inizializzazione del PageRank
    double *pagerank = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        pagerank[i] = 1.0 / N;
    }

    // Creazione dei thread per il calcolo del PageRank
    pthread_t threads[4];
    thread_data tdata[4];
    int step = N / 4;

    for (int i = 0; i < 4; i++) {
        tdata[i].g = g;
        tdata[i].pagerank = pagerank;
        tdata[i].start = i * step;
        tdata[i].end = (i == 3) ? N : (i + 1) * step;
        pthread_create(&threads[i], NULL, calcola_pagerank, &tdata[i]);
    }

    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL); // Sincronizza i thread
    }

    // Stampa i risultati
    for (int i = 0; i < N; i++) {
        printf("Node %d: PageRank = %.6f\n", i, pagerank[i]);
    }

    free(pagerank);
    return 0;
}

//output 2

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#define MAX_THREADS 8
#define MAX_BUFFER_SIZE 100

// Struttura per rappresentare il grafo con lista di adiacenza
typedef struct {
    int N;           // numero di nodi
    int *out;        // array con il numero di archi uscenti per ogni nodo
    int **in;        // lista di adiacenza per i nodi entranti
    int *in_sizes;   // array con la dimensione di ogni lista di adiacenza
} grafo;

// Buffer per il produttore-consumatore
typedef struct {
    int i, j;
} arco;

// Struttura per buffer dinamico
arco buffer[MAX_BUFFER_SIZE];
int buffer_in = 0, buffer_out = 0;
sem_t empty, full;
pthread_mutex_t buffer_mutex = PTHREAD_MUTEX_INITIALIZER;

// Variabili globali
int current_iteration = 0;
double *pagerank_values;

// Parametri di esecuzione
double damping_factor = 0.85;
double epsilon = 1e-6;
int max_iterations = 100;
int top_k = 3;

// Thread per la gestione dei segnali
pthread_t signal_thread;
pthread_mutex_t pagerank_mutex = PTHREAD_MUTEX_INITIALIZER;
int sigusr1_received = 0;

void add_to_buffer(int i, int j) {
    sem_wait(&empty);
    pthread_mutex_lock(&buffer_mutex);
    buffer[buffer_in].i = i;
    buffer[buffer_in].j = j;
    buffer_in = (buffer_in + 1) % MAX_BUFFER_SIZE;
    pthread_mutex_unlock(&buffer_mutex);
    sem_post(&full);
}

arco remove_from_buffer() {
    sem_wait(&full);
    pthread_mutex_lock(&buffer_mutex);
    arco item = buffer[buffer_out];
    buffer_out = (buffer_out + 1) % MAX_BUFFER_SIZE;
    pthread_mutex_unlock(&buffer_mutex);
    sem_post(&empty);
    return item;
}

// Funzione di gestione del segnale SIGUSR1
void* signal_handler(void* arg) {
    while (1) {
        pause();
        if (sigusr1_received) {
            pthread_mutex_lock(&pagerank_mutex);
            printf("Segnale SIGUSR1 ricevuto - Iterazione: %d\n", current_iteration);
            int max_node = 0;
            double max_pagerank = pagerank_values[0];
            for (int i = 1; i < ((grafo*)arg)->N; i++) {
                if (pagerank_values[i] > max_pagerank) {
                    max_node = i;
                    max_pagerank = pagerank_values[i];
                }
            }
            printf("Nodo con il massimo PageRank: %d, valore: %f\n", max_node, max_pagerank);
            pthread_mutex_unlock(&pagerank_mutex);
            sigusr1_received = 0;
        }
    }
    return NULL;
}

// Funzione per il calcolo del PageRank
double* pagerank(grafo* g, double d, double eps, int maxiter, int* numiter) {
    double *x = (double*)malloc(g->N * sizeof(double));
    double *xnext = (double*)malloc(g->N * sizeof(double));
    
    for (int i = 0; i < g->N; i++) {
        x[i] = 1.0 / g->N;
    }

    int iter = 0;
    double diff;
    
    do {
        double dead_end_contrib = 0;
        for (int i = 0; i < g->N; i++) {
            if (g->out[i] == 0) {
                dead_end_contrib += d * x[i] / g->N;
            }
        }
        
        for (int j = 0; j < g->N; j++) {
            xnext[j] = (1 - d) / g->N + dead_end_contrib;
            for (int k = 0; k < g->in_sizes[j]; k++) {
                int i = g->in[j][k];
                xnext[j] += d * x[i] / g->out[i];
            }
        }
        
        diff = 0;
        for (int i = 0; i < g->N; i++) {
            diff += fabs(xnext[i] - x[i]);
            x[i] = xnext[i];
        }
        
        iter++;
    } while (diff > eps && iter < maxiter);
    
    *numiter = iter;
    free(xnext);
    return x;
}

// Funzione per la lettura del file e gestione del buffer
void* thread_consumer(void* arg) {
    grafo* g = (grafo*)arg;
    
    while (1) {
        arco arc = remove_from_buffer();
        if (arc.i == -1 && arc.j == -1) break;  // Segnale per terminare
        
        // Gestione dell'arco nel grafo
        if (arc.i != arc.j) {
            // Aggiunta a lista di adiacenza, evitando duplicati
            int found = 0;
            for (int k = 0; k < g->in_sizes[arc.j]; k++) {
                if (g->in[arc.j][k] == arc.i) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                g->in[arc.j] = realloc(g->in[arc.j], (g->in_sizes[arc.j] + 1) * sizeof(int));
                g->in[arc.j][g->in_sizes[arc.j]] = arc.i;
                g->in_sizes[arc.j]++;
                g->out[arc.i]++;
            }
        }
    }
    
    return NULL;
}

// Funzione principale
int main(int argc, char** argv) {
    sem_init(&empty, 0, MAX_BUFFER_SIZE);
    sem_init(&full, 0, 0);

    // Inizializzazione grafo (da leggere successivamente dal file)
    grafo g;
    g.N = 9; // esempio, leggi dal file
    g.out = (int*)calloc(g.N, sizeof(int));
    g.in = (int**)calloc(g.N, sizeof(int*));
    g.in_sizes = (int*)calloc(g.N, sizeof(int));

    // Creazione dei thread per il buffer produttore-consumatore
    pthread_t consumers[MAX_THREADS];
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_create(&consumers[i], NULL, thread_consumer, (void*)&g);
    }

    // Lettura del file e aggiunta al buffer
    FILE* fp = fopen("9nodi.mtx", "r");
    if (!fp) {
        perror("Errore durante l'apertura del file");
        return 1;
    }

    // Esempio di lettura semplificata (da estendere)
    int i, j;
    while (fscanf(fp, "%d %d", &i, &j) != EOF) {
        add_to_buffer(i - 1, j - 1);
    }
    fclose(fp);

    // Segnale di terminazione ai thread
    for (int i = 0; i < MAX_THREADS; i++) {
        add_to_buffer(-1, -1);
    }

    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(consumers[i], NULL);
    }

    // Calcolo del PageRank
    int num_iter = 0;
    pagerank_values = pagerank(&g, damping_factor, epsilon, max_iterations, &num_iter);

    // Stampa del risultato
    printf("PageRank calcolato in %d iterazioni\n", num_iter);
    
    // Cleanup
    free(g.out);
    for (int i = 0; i < g.N; i++) {
        free(g.in[i]);
    }
    free(g.in);
    free(g.in_sizes);
    free(pagerank_values);
    
    sem_destroy(&empty);
    sem_destroy(&full);
    
    return 0;
}

