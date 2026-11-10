import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;
import java.util.stream.*;
import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.BorderLayout;
import java.awt.ComponentOrientation;
import java.awt.Font;
import safar.basic.morphology.stemmer.factory.StemmerFactory;
import safar.basic.morphology.stemmer.interfaces.IStemmer;
import safar.basic.morphology.stemmer.model.StemmerAnalysis;
import safar.basic.morphology.stemmer.model.WordStemmerAnalysis;

public class Main {

    // ====== A MODIFIER : chemin du dossier contenant les .txt ======
    static final String CORPUS_DIR = "corpus";
    static final String OUT_DIR = "resultats";
    static final int MIN_DF = 5;          // un terme doit apparaitre dans au moins 5 documents
    static final boolean STEMMING = true; // racinisation avec SAFAR (Light10)

    static final Set<String> STOP = new HashSet<>();
    static {
        String s = "من في على الى إلى عن مع هذا هذه ذلك تلك التي الذي الذين التى هو هي هم هن "
                 + "ان أن إن أنه انه أنها انها كان كانت يكون تكون لم لن لا ما ماذا قد كل بعد قبل بين "
                 + "حيث اذا إذا او أو ثم كما لما لكن غير عند عليه عليها عليهم له لها لهم به بها بهم "
                 + "فيه فيها منه منها وقد وان وأن وفي ومن وعلى وإلى والتي والذي فإن فان هؤلاء "
                 + "أي اي بما إذ اذ حتى سواء أمام ضد دون خلال وفق بموجب يوم عدد رقم";
        for (String w : s.split("\\s+")) STOP.add(normalize(w));
    }

    static String normalize(String t) {
        t = t.replaceAll("[\\u064B-\\u0652\\u0640]", "");
        t = t.replaceAll("[إأآ]", "ا");
        return t.replace('ى', 'ي').replace('ة', 'ه');
    }

    // Stemmer Light10 de la bibliotheque SAFAR (+ cache pour aller plus vite)
    static final IStemmer SAFAR_STEMMER = StemmerFactory.getLight10Implementation();
    static final Map<String, String> CACHE = new HashMap<>();

    static String lightStem(String w) {
        if (!STEMMING) return w;
        String r = CACHE.get(w);
        if (r != null) return r;
        r = w;
        try {
            List<WordStemmerAnalysis> res = SAFAR_STEMMER.stem(w);
            if (!res.isEmpty()) {
                for (StemmerAnalysis a : res.get(0).getListStemmerAnalysis()) {
                    if ("STEM".equals(a.getType()) && a.getMorpheme() != null && !a.getMorpheme().isEmpty()) {
                        r = normalize(a.getMorpheme());
                        break;
                    }
                }
            }
        } catch (Exception e) { /* on garde le mot tel quel */ }
        CACHE.put(w, r);
        return r;
    }

    static List<String> preprocess(String text) {
        List<String> out = new ArrayList<>();
        Matcher m = Pattern.compile("[\\u0621-\\u064A]+").matcher(text);
        while (m.find()) {
            String w = normalize(m.group());
            if (w.length() <= 2 || STOP.contains(w)) continue;
            String st = lightStem(w);
            if (st.length() >= 3 && !STOP.contains(st)) out.add(st);
        }
        return out;
    }

    // ====== DONNEES DU MOTEUR (chargees une seule fois) ======
    static double[][] M;
    static double[] IDF;
    static Map<String, Integer> POS = new HashMap<>();
    static List<String> NOMS = new ArrayList<>();
    static List<String> TEXTES = new ArrayList<>();
    static final int TOP = 20;

    // ====== 1) INDEXATION : lecture du corpus, matrices occurrences + TF-IDF ======
    static void indexer(String dir) throws IOException {
        List<Path> files;
        try (Stream<Path> st = Files.walk(Paths.get(dir))) {
            files = st.filter(p -> p.toString().toLowerCase().endsWith(".txt"))
                      .sorted(Comparator.comparingInt(Main::num)).collect(Collectors.toList());
        }
        System.out.println(files.size() + " documents trouves");

        List<Map<String, Integer>> counts = new ArrayList<>();
        Map<String, Integer> df = new HashMap<>();
        for (Path p : files) {
            String text = new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
            Map<String, Integer> c = new HashMap<>();
            for (String w : preprocess(text)) c.merge(w, 1, Integer::sum);
            counts.add(c);
            for (String w : c.keySet()) df.merge(w, 1, Integer::sum);
            NOMS.add(p.getFileName().toString());
            TEXTES.add(text.trim());
        }
        List<String> vocab = df.entrySet().stream().filter(e -> e.getValue() >= MIN_DF)
                .map(Map.Entry::getKey).sorted().collect(Collectors.toList());
        System.out.println(vocab.size() + " termes retenus");

        int N = files.size();
        int[] docLen = new int[N];
        for (int i = 0; i < N; i++) {
            int s = 0;
            for (String t : vocab) s += counts.get(i).getOrDefault(t, 0);
            docLen[i] = Math.max(s, 1);
        }

        IDF = new double[vocab.size()];
        for (int j = 0; j < vocab.size(); j++) {
            POS.put(vocab.get(j), j);
            IDF[j] = Math.log((double) N / df.get(vocab.get(j)));
        }
        M = new double[N][vocab.size()];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < vocab.size(); j++)
                M[i][j] = ((double) counts.get(i).getOrDefault(vocab.get(j), 0) / docLen[i]) * IDF[j];

        // sauvegarde des deux matrices en CSV
        Files.createDirectories(Paths.get(OUT_DIR));
        try (PrintWriter occ = writer(OUT_DIR + "/matrice_occurrences.csv");
             PrintWriter tfidf = writer(OUT_DIR + "/matrice_tfidf.csv")) {
            String header = "document," + String.join(",", vocab);
            occ.println(header);
            tfidf.println(header);
            for (int i = 0; i < N; i++) {
                StringBuilder a = new StringBuilder(NOMS.get(i));
                StringBuilder b = new StringBuilder(NOMS.get(i));
                for (int j = 0; j < vocab.size(); j++) {
                    a.append(',').append(counts.get(i).getOrDefault(vocab.get(j), 0));
                    b.append(',').append(String.format(Locale.US, "%.6f", M[i][j]));
                }
                occ.println(a);
                tfidf.println(b);
            }
        }
        System.out.println("Matrices enregistrees dans '" + OUT_DIR + "'");
    }

    // ====== 2) RECHERCHE : vecteur de la requete + cosinus ======
    // retourne la liste des indices de documents classes (score > 0), scores dans 'scores'
    static List<Integer> rechercher(String requete, List<String> motsOut, Map<Integer, Double> scores) {
        List<String> mots = preprocess(requete);
        motsOut.addAll(mots);
        double[] q = new double[IDF.length];
        boolean connu = false;
        for (String w : mots) {
            Integer j = POS.get(w);
            if (j != null) { q[j] += 1.0 / mots.size(); connu = true; }
        }
        if (!connu) return new ArrayList<>();
        for (int j = 0; j < q.length; j++) q[j] *= IDF[j];
        double nq = norme(q);
        List<Integer> ordre = new ArrayList<>();
        for (int i = 0; i < M.length; i++) {
            double nd = norme(M[i]), dot = 0;
            for (int j = 0; j < q.length; j++) dot += q[j] * M[i][j];
            double sc = (nd == 0 || nq == 0) ? 0 : dot / (nq * nd);
            if (sc > 0) { scores.put(i, sc); ordre.add(i); }
        }
        ordre.sort((a, b) -> Double.compare(scores.get(b), scores.get(a)));
        return ordre.size() > TOP ? ordre.subList(0, TOP) : ordre;
    }

    // ====== 3) INTERFACE GRAPHIQUE ======
    static void creerFenetre() {
        Font fontAr = new Font("Arial", Font.PLAIN, 18);

        JFrame f = new JFrame("Moteur de recherche - Chambre immobiliere (TF-IDF + cosinus)");
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.setSize(1000, 700);
        f.setLocationRelativeTo(null);

        // --- barre de recherche ---
        JTextField champ = new JTextField();
        champ.setFont(fontAr);
        champ.setComponentOrientation(ComponentOrientation.RIGHT_TO_LEFT);
        JButton bouton = new JButton("Rechercher");
        bouton.setFont(new Font("Arial", Font.BOLD, 14));
        JLabel statut = new JLabel("Tapez votre requete en arabe puis appuyez sur Entree.");
        JPanel haut = new JPanel(new BorderLayout(8, 8));
        haut.setBorder(BorderFactory.createEmptyBorder(10, 10, 5, 10));
        haut.add(bouton, BorderLayout.WEST);
        haut.add(champ, BorderLayout.CENTER);
        haut.add(statut, BorderLayout.SOUTH);

        // --- tableau des resultats ---
        DefaultTableModel modele = new DefaultTableModel(new Object[]{"Rang", "Document", "Score", "Apercu"}, 0) {
            public boolean isCellEditable(int r, int c) { return false; }
        };
        JTable table = new JTable(modele);
        table.setFont(new Font("Arial", Font.PLAIN, 14));
        table.setRowHeight(24);
        table.getColumnModel().getColumn(0).setMaxWidth(50);
        table.getColumnModel().getColumn(1).setMaxWidth(90);
        table.getColumnModel().getColumn(2).setMaxWidth(80);
        table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        // --- texte complet du document selectionne ---
        JTextArea texte = new JTextArea();
        texte.setFont(fontAr);
        texte.setEditable(false);
        texte.setLineWrap(true);
        texte.setWrapStyleWord(true);
        texte.setComponentOrientation(ComponentOrientation.RIGHT_TO_LEFT);

        JSplitPane split = new JSplitPane(JSplitPane.VERTICAL_SPLIT,
                new JScrollPane(table), new JScrollPane(texte));
        split.setResizeWeight(0.4);

        final List<Integer> affiches = new ArrayList<>();

        Runnable lancer = () -> {
            String requete = champ.getText().trim();
            if (requete.isEmpty()) return;
            modele.setRowCount(0);
            affiches.clear();
            texte.setText("");
            List<String> mots = new ArrayList<>();
            Map<Integer, Double> scores = new HashMap<>();
            List<Integer> res = rechercher(requete, mots, scores);
            if (res.isEmpty()) {
                statut.setText("Aucun resultat (termes analyses : " + mots + ")");
                return;
            }
            int rang = 1;
            for (int i : res) {
                String t = TEXTES.get(i).replaceAll("\\s+", " ");
                String apercu = t.length() > 110 ? t.substring(0, 110) + "..." : t;
                modele.addRow(new Object[]{rang++, NOMS.get(i), String.format(Locale.US, "%.4f", scores.get(i)), apercu});
                affiches.add(i);
            }
            statut.setText(res.size() + " resultat(s) - termes analyses : " + mots);
            table.setRowSelectionInterval(0, 0);
        };
        bouton.addActionListener(e -> lancer.run());
        champ.addActionListener(e -> lancer.run());

        table.getSelectionModel().addListSelectionListener(e -> {
            int r = table.getSelectedRow();
            if (!e.getValueIsAdjusting() && r >= 0 && r < affiches.size()) {
                texte.setText(TEXTES.get(affiches.get(r)));
                texte.setCaretPosition(0);
            }
        });

        f.setLayout(new BorderLayout());
        f.add(haut, BorderLayout.NORTH);
        f.add(split, BorderLayout.CENTER);
        f.setVisible(true);
        champ.requestFocusInWindow();
    }

    public static void main(String[] args) throws IOException {
        String dir = args.length > 0 ? args[0] : CORPUS_DIR;
        indexer(dir);
        SwingUtilities.invokeLater(Main::creerFenetre);
    }

    static double norme(double[] v) {
        double s = 0;
        for (double x : v) s += x * x;
        return Math.sqrt(s);
    }

    static int num(Path p) {
        String d = p.getFileName().toString().replaceAll("\\D", "");
        return d.isEmpty() ? 0 : Integer.parseInt(d);
    }

    static PrintWriter writer(String f) throws IOException {
        OutputStream os = new FileOutputStream(f);
        os.write(new byte[]{(byte) 0xEF, (byte) 0xBB, (byte) 0xBF}); // BOM pour Excel
        return new PrintWriter(new OutputStreamWriter(os, StandardCharsets.UTF_8));
    }
}