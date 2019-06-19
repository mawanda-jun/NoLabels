# Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzle

## Idea
Self-supervised learning tramite puzzle di Jingsaw:
- vengono presi 9 pezzi da un'immagine contenente un oggetto;
- i pezzi vengono mescolati;
- ogni pezzo può essere indentificato come una parte dell'oggetto;
- i pezzi che sono ambigui e che non vengono identificati come ciò possono essere capito quando tutti i pezzi del puzzle sono al loro posto;
- Challenge: identificare la posizione del pezzo centrale e dei due pezzi in alto a sinistra.

La risoluzione di un puzzle di Jigsaw può essere utilizzata per insegnare ad un sistema che un oggetto è composto da più parti e da che parti sono.

L'esperimento è composto da due parti:
- pre-training -> learning tramite Jigsaw puzzle
- fine-tuning -> utilizzare il risultato del pre-training per applicarlo a un problema di classificazione di immagini (questo è il punto: cosa si intende utilizzare il risultato del pre-training? Da quello che ho capito si usa la rete allenata con Jigsaw, si usano i pesi, li si spiaccicano dentro alla rete e da lì si fa un training minore, quindi inferenza.)

## Context Free Network
Usata per generare e risolvere un puzzle.
Ogni immagine viene ritaglita a 225 x 225 pixel, divisa in una griglia 3 x 3 e viene preso casualmente un quadrato di 64 x 64 pizel all'interno di ogni cella dell'immagine ritagliata. L'obiettivo è quello di predirre l'indice della permutazione scelta da un vettore che contiene varie possibili permutazioni.

 9 pezzi -> 9! = 362,880 posibili permutazioni -> il set di permutazioni è un fattore importante sulle performance di rappresentazione di cosa apprende la rete.

Le varie permutazioni di puzzle devono essere abbastanza diverse tra loro -> vengono scelte combinazioni che abbiano una sufficentemente larga distanza di Hamming tra loro (se ne scelgono 69).

Per evitare shortcuts:
-  inserimento di un gap tra i pezzi dei puzzle (gap random tra 0 e 22 pixel)
- immagini in scala di grigio
- normalizzazione della deviazione standard separatamente per ogni chunk)

E' stato utilizzato Caffe per creare i pezzi delle immagini e le permutazioni durante il training -> questo ha permesso di tenere il dataset piccolo (e il training efficiente? Non è detto)

## Esperimenti
Vanno tenuti in considerazione alcuni fattori per gli insieme di permutazioni far sì che la predizione della soluzione del puzzle non risulti impossibile:
- cardinalità
- media della distanza di Hamming
- distanza minima di Hamming

## Dubbi

- Non mi è molto chiaro perchè hanno utilizzato due dataset (PASCAL VOC 2007 e Image Net) e come sono stati utilizzati nelle varie parti degli esperimenti. Cioè hanno utilizzato prima Pascal voc 2007 perchè è più piccolo e ha meno categorie rispetto a ImageNet?

    Ballan vorrebbe solo che utilizzassimo il primo oppure Caltech-101 / Caltech-256?

- Le immagini vengono prima tagliate a 256 x 256 pixel e poi viene estratto un quadrato di 225 x 255, giusto? Dovrebbero essere immagini 256x256 e seleziona array 0-255 teoricamente.







