import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;
import java.util.stream.*;
import safar.basic.morphology.stemmer.factory.StemmerFactory;
import safar.basic.morphology.stemmer.interfaces.IStemmer;
import safar.basic.morphology.stemmer.model.StemmerAnalysis;
import safar.basic.morphology.stemmer.model.WordStemmerAnalysis;

/**
 * TP2 - Classification de textes par Naive Bayes (cours "Apprentissage supervise").
 * Classe (-) = chambre immobiliere ; classe (+) = statut personnel et heritage.
 *  P(c)    = nb documents de c / nb documents d'apprentissage
 *  P(w|c)  = (nk + 1) / (n + m)    (lissage de Laplace)
 *     n  = nombre total de mots dans les documents de la classe c
 *     nk = nombre de fois ou le mot w apparait dans la classe c
 *     m  = nombre de mots distincts du vocabulaire d'apprentissage
 */
public class Classification {

    // ====== PARAMETRES ======
    static final String DOSSIER = "corp   us_classes";
    static final String CLASSE_MOINS = "immobilier";        // sous-dossier de la classe (-)
    static final String CLASSE_PLUS  = "statut_personnel";  // sous-dossier de la classe (+) = la plus grande base
    static final int TEST_PLUS = 51, TEST_MOINS = 34;       // test cache : 34/85 = 2/5 et 51/85 = 3/5
    static final int TRAIN_MOINS = 320, TRAIN_PLUS = 480;   // apprentissage : 320/800 = 2/5 et 480/800 = 3/5
    static final long GRAINE = 42;                          // meme tirage a chaque execution

    // ====== PRETRAITEMENT (normalisation, mots vides, stemming SAFAR Light10) ======
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

    static final IStemmer SAFAR_STEMMER = StemmerFactory.getLight10Implementation();
    static final Map<String, String> CACHE = new HashMap<>();

    static String stem(String w) {
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
            String st = stem(w);
            if (st.length() >= 3 && !STOP.contains(st)) out.add(st);
        }
        return out;
    }

    // ====== UN DOCUMENT = (nom, classe, mots) ======
    static class Doc {
        String nom, classe;
        List<String> mots;
        Doc(String nom, String classe, List<String> mots) { this.nom = nom; this.classe = classe; this.mots = mots; }
    }

    // ====== MODELE APPRIS (rempli uniquement avec les documents d'apprentissage) ======
    static Map<String, Map<String, Integer>> NK = new HashMap<>();   // classe -> (mot -> nk)
    static Map<String, Integer> N = new HashMap<>();                 // classe -> n
    static Set<String> VOCAB = new HashSet<>();                      // mots d'apprentissage
    static Map<String, Double> PRIOR = new HashMap<>();              // classe -> P(c)
    static Map<String, Map<String, Float>> PROB = new HashMap<>();   // classe -> (mot -> P(w|c))

    static int calculerM() { return VOCAB.size(); }                                       // m
    static int calculerN(String classe) { return N.get(classe); }                         // n
    static int calculerNk(String classe, String mot) { return NK.get(classe).getOrDefault(mot, 0); }  // nk

    // lecture d'une classe : tous les .txt du sous-dossier ; les textes identiques (deja vus) sont ignores
    static List<Doc> lireClasse(String sousDossier, Set<String> dejaVus) throws IOException {
        List<Doc> docs = new ArrayList<>();
        try (Stream<Path> st = Files.walk(Paths.get(DOSSIER, sousDossier))) {
            List<Path> files = st.filter(p -> p.toString().toLowerCase().endsWith(".txt"))
                                 .sorted().collect(Collectors.toList());
            for (Path p : files) {
                String texte = new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
                String empreinte = texte.replaceAll("\\s+", " ").trim();
                if (!dejaVus.add(empreinte)) continue;          // doublon : on ne le garde qu'une fois
                docs.add(new Doc(sousDossier + "/" + p.getFileName(), sousDossier, preprocess(texte)));
            }
        }
        return docs;
    }

    // apprentissage : calcule n, nk, m, P(c) et P(w|c) avec la liste d'apprentissage SEULEMENT
    static void entrainer(List<Doc> train) {
        for (Doc d : train) {
            NK.computeIfAbsent(d.classe, k -> new HashMap<>());
            N.merge(d.classe, d.mots.size(), Integer::sum);
            for (String w : d.mots) {
                NK.get(d.classe).merge(w, 1, Integer::sum);
                VOCAB.add(w);
            }
        }
        int m = calculerM();
        for (String c : NK.keySet()) {
            long nbDocs = train.stream().filter(d -> d.classe.equals(c)).count();
            PRIOR.put(c, (double) nbDocs / train.size());
            Map<String, Float> p = new HashMap<>();
            int n = calculerN(c);
            for (String w : VOCAB)
                p.put(w, (float) (calculerNk(c, w) + 1) / (n + m));
            PROB.put(c, p);
        }
    }

    // classification : la sortie est UNIQUEMENT le nom de la classe (String)
    static String classer(List<String> mots) {
        String meilleure = null;
        double meilleurScore = Double.NEGATIVE_INFINITY;
        for (String c : PROB.keySet()) {
            double score = Math.log(PRIOR.get(c));
            for (String w : mots) {
                Float p = PROB.get(c).get(w);
                if (p != null) score += Math.log(p);    // mot inconnu de l'apprentissage : ignore
            }
            if (score > meilleurScore) { meilleurScore = score; meilleure = c; }
        }
        return meilleure;
    }

    static String classerTexte(String texte) { return classer(preprocess(texte)); }

    public static void main(String[] args) throws IOException {
        // 1) toutes les classes dans UNE seule liste : d1-c1, d2-c1, d3-c2, ...
        Set<String> dejaVus = new HashSet<>();
        List<Doc> moins = lireClasse(CLASSE_MOINS, dejaVus);
        List<Doc> plus = lireClasse(CLASSE_PLUS, dejaVus);
        System.out.println(moins.size() + " documents (" + CLASSE_MOINS + "), " + plus.size() + " documents (" + CLASSE_PLUS + ")");
        Collections.shuffle(moins, new Random(GRAINE));
        Collections.shuffle(plus, new Random(GRAINE));

        // 2) TEST CACHE : tire en premier, puis RETIRE des listes d'origine
        List<Doc> test = new ArrayList<>();
        test.addAll(moins.subList(0, TEST_MOINS));
        test.addAll(plus.subList(0, TEST_PLUS));
        List<Doc> resteMoins = new ArrayList<>(moins.subList(TEST_MOINS, moins.size()));
        List<Doc> restePlus = new ArrayList<>(plus.subList(TEST_PLUS, plus.size()));

        // 3) APPRENTISSAGE : tire uniquement parmi les documents restants
        List<Doc> train = new ArrayList<>();
        train.addAll(resteMoins.subList(0, TRAIN_MOINS));
        train.addAll(restePlus.subList(0, TRAIN_PLUS));

        // verification : aucun document de test ne se trouve dans l'apprentissage
        Set<String> nomsTest = test.stream().map(d -> d.nom).collect(Collectors.toSet());
        long chevauchement = train.stream().filter(d -> nomsTest.contains(d.nom)).count();
        System.out.println("Apprentissage : " + train.size() + " documents, test : " + test.size()
                + " documents, chevauchement = " + chevauchement);
        if (chevauchement != 0) throw new IllegalStateException("Des documents de test sont dans l'apprentissage !");

        // 4) apprentissage (le test n'est pas encore utilise)
        entrainer(train);
        System.out.println("m = " + calculerM());
        for (String c : new String[]{CLASSE_MOINS, CLASSE_PLUS})
            System.out.printf(Locale.US, "classe %-17s n = %d   P(c) = %.2f%n", c, calculerN(c), PRIOR.get(c));

        // 5) test sur les documents caches
        int tpM = 0, fpM = 0, fnM = 0, tpP = 0, fpP = 0, fnP = 0, ok = 0;
        System.out.println("\nDocument                    vraie classe      classe predite");
        for (Doc d : test) {
            String pred = classer(d.mots);
            String signe = pred.equals(d.classe) ? "" : "   <-- erreur";
            System.out.printf("%-27s %-17s %s%s%n", d.nom, d.classe, pred, signe);
            if (pred.equals(d.classe)) ok++;
            if (d.classe.equals(CLASSE_MOINS)) { if (pred.equals(CLASSE_MOINS)) tpM++; else fnM++; }
            else { if (pred.equals(CLASSE_PLUS)) tpP++; else fnP++; }
            if (pred.equals(CLASSE_MOINS) && !d.classe.equals(CLASSE_MOINS)) fpM++;
            if (pred.equals(CLASSE_PLUS) && !d.classe.equals(CLASSE_PLUS)) fpP++;
        }

        // 6) precision, rappel, F-score
        System.out.printf(Locale.US, "%nExactitude (accuracy) : %d/%d = %.2f%n", ok, test.size(), (double) ok / test.size());
        afficher(CLASSE_MOINS, tpM, fpM, fnM);
        afficher(CLASSE_PLUS, tpP, fpP, fnP);
    }

    static void afficher(String classe, int tp, int fp, int fn) {
        double precision = tp + fp == 0 ? 0 : (double) tp / (tp + fp);
        double rappel = tp + fn == 0 ? 0 : (double) tp / (tp + fn);
        double f = precision + rappel == 0 ? 0 : 2 * precision * rappel / (precision + rappel);
        System.out.printf(Locale.US, "Classe %-17s precision = %.2f   rappel = %.2f   F-score = %.2f   (VP=%d FP=%d FN=%d)%n",
                classe, precision, rappel, f, tp, fp, fn);
    }
}