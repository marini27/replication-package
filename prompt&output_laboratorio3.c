//ARCHITECTURAL+RECIPE
/*puoi fare una ricetta dettagliata che implementi la costruzione di questa architettura client-server? La ricetta viene seguita da un programmatore che vuole sviluppare il progetto, quindi utilizza un linguaggio tecnico, preciso e dettagliato.*/

//output 1
//server
import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.*;

public class WordleServer {
    private static final int PORT = 12345;
    private static final Map<String, User> registeredUsers = new ConcurrentHashMap<>();
    private static String secretWord = "HELLO";
    private static ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

    public static void main(String[] args) {
        System.out.println("Wordle Server is running...");
        scheduler.scheduleAtFixedRate(WordleServer::updateSecretWord, 0, 24, TimeUnit.HOURS);

        try (ServerSocket serverSocket = new ServerSocket(PORT)) {
            while (true) {
                Socket clientSocket = serverSocket.accept();
                new Thread(new ClientHandler(clientSocket)).start();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void updateSecretWord() {
        // Logica per aggiornare la parola segreta
        secretWord = "NEWWORD"; // Esempio di aggiornamento
        System.out.println("New secret word: " + secretWord);
    }

    private static class ClientHandler implements Runnable {
        private Socket socket;

        ClientHandler(Socket socket) {
            this.socket = socket;
        }

        @Override
        public void run() {
            try (
                BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                PrintWriter out = new PrintWriter(socket.getOutputStream(), true)
            ) {
                String request;
                while ((request = in.readLine()) != null) {
                    String[] tokens = request.split(" ");
                    String command = tokens[0];

                    switch (command.toLowerCase()) {
                        case "register":
                            String username = tokens[1];
                            String password = tokens[2];
                            if (registerUser(username, password)) {
                                out.println("Registration successful");
                            } else {
                                out.println("Username already taken");
                            }
                            break;
                        case "login":
                            username = tokens[1];
                            password = tokens[2];
                            if (loginUser(username, password)) {
                                out.println("Login successful");
                            } else {
                                out.println("Invalid credentials or already logged in");
                            }
                            break;
                        case "playwordle":
                            out.println("The secret word is ready. Start guessing!");
                            break;
                        default:
                            out.println("Unknown command");
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        private boolean registerUser(String username, String password) {
            if (registeredUsers.containsKey(username)) {
                return false;
            }
            registeredUsers.put(username, new User(username, password));
            return true;
        }

        private boolean loginUser(String username, String password) {
            User user = registeredUsers.get(username);
            if (user != null && user.getPassword().equals(password) && !user.isLoggedIn()) {
                user.setLoggedIn(true);
                return true;
            }
            return false;
        }
    }

    static class User {
        private final String username;
        private final String password;
        private boolean loggedIn;

        User(String username, String password) {
            this.username = username;
            this.password = password;
            this.loggedIn = false;
        }

        String getPassword() {
            return password;
        }

        boolean isLoggedIn() {
            return loggedIn;
        }

        void setLoggedIn(boolean loggedIn) {
            this.loggedIn = loggedIn;
        }
    }
}

//client
import java.io.*;
import java.net.*;
import java.util.Scanner;

public class WordleClient {
    private static final String SERVER_ADDRESS = "localhost";
    private static final int SERVER_PORT = 12345;

    public static void main(String[] args) {
        try (Socket socket = new Socket(SERVER_ADDRESS, SERVER_PORT);
             BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
             PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
             Scanner scanner = new Scanner(System.in)) {

            System.out.println("Connected to Wordle Server");

            while (true) {
                System.out.print("Enter command: ");
                String command = scanner.nextLine();
                out.println(command);

                String response = in.readLine();
                if (response == null) {
                    break;
                }
                System.out.println("Server: " + response);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}


//output 2
//server
import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.*;

public class WordleServer {
    private static final int PORT = 12345;
    private static Map<String, String> users = new HashMap<>(); // username -> password
    private static Map<String, Integer> userStatistics = new HashMap<>(); // username -> games won
    private static String secretWord = "apple"; // parola segreta di esempio

    public static void main(String[] args) throws IOException {
        // Crea il socket del server
        ServerSocket serverSocket = new ServerSocket(PORT);
        System.out.println("Server in ascolto sulla porta " + PORT);

        // Usa un thread pool per gestire più client
        ExecutorService threadPool = Executors.newFixedThreadPool(10);

        while (true) {
            Socket clientSocket = serverSocket.accept();
            System.out.println("Nuovo client connesso");
            threadPool.submit(new ClientHandler(clientSocket));
        }
    }

    // Gestore delle richieste dei client
    private static class ClientHandler implements Runnable {
        private Socket clientSocket;
        private BufferedReader in;
        private PrintWriter out;

        public ClientHandler(Socket socket) {
            this.clientSocket = socket;
        }

        public void run() {
            try {
                in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
                out = new PrintWriter(clientSocket.getOutputStream(), true);

                String request;
                while ((request = in.readLine()) != null) {
                    System.out.println("Richiesta ricevuta: " + request);
                    handleRequest(request);
                }
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                try {
                    clientSocket.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        private void handleRequest(String request) {
            String[] parts = request.split(" ");
            String command = parts[0];

            switch (command) {
                case "register":
                    registerUser(parts[1], parts[2]);
                    break;
                case "login":
                    loginUser(parts[1], parts[2]);
                    break;
                case "playWordle":
                    startGame(parts[1]);
                    break;
                case "sendWord":
                    checkWord(parts[1], parts[2]);
                    break;
                case "sendMeStatistics":
                    sendStatistics(parts[1]);
                    break;
                case "share":
                    shareResults(parts[1], parts[2]);
                    break;
                default:
                    out.println("Comando non riconosciuto");
            }
        }

        private void registerUser(String username, String password) {
            if (users.containsKey(username)) {
                out.println("Username già in uso");
            } else {
                users.put(username, password);
                out.println("Registrazione avvenuta con successo");
            }
        }

        private void loginUser(String username, String password) {
            if (users.containsKey(username) && users.get(username).equals(password)) {
                out.println("Login effettuato con successo");
            } else {
                out.println("Login fallito");
            }
        }

        private void startGame(String username) {
            out.println("Inizia a indovinare la parola segreta");
        }

        private void checkWord(String username, String guessedWord) {
            if (guessedWord.equals(secretWord)) {
                out.println("Hai indovinato la parola!");
                userStatistics.put(username, userStatistics.getOrDefault(username, 0) + 1);
            } else {
                out.println("Parola sbagliata, riprova");
            }
        }

        private void sendStatistics(String username) {
            int gamesWon = userStatistics.getOrDefault(username, 0);
            out.println("Statistiche per " + username + ": partite vinte = " + gamesWon);
        }

        private void shareResults(String username, String result) {
            out.println("Risultato condiviso con gli altri giocatori: " + result);
            // Invia messaggio multicast (simulato per ora)
        }
    }
}

//client
import java.io.*;
import java.net.*;
import java.util.*;

public class WordleClient {
    private static final String SERVER_ADDRESS = "127.0.0.1";
    private static final int SERVER_PORT = 12345;

    public static void main(String[] args) throws IOException {
        // Connessione al server
        Socket socket = new Socket(SERVER_ADDRESS, SERVER_PORT);
        BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        PrintWriter out = new PrintWriter(socket.getOutputStream(), true);

        // Scanner per l'input da tastiera
        Scanner scanner = new Scanner(System.in);
        String command;

        // Loop principale per l'interazione con l'utente
        while (true) {
            System.out.println("Inserisci comando:");
            command = scanner.nextLine();

            if (command.equals("exit")) {
                break;
            }

            out.println(command);
            String response = in.readLine();
            System.out.println("Risposta dal server: " + response);
        }

        // Chiudi le risorse
        socket.close();
        scanner.close();
    }
}



//GAME PATTERN
//sviluppa il gioco descritto nel seguente pdf. Restituisci il codice java.

//output 1
// WordleClient.java
import java.io.*;
import java.net.*;
import java.util.Scanner;

public class WordleClient {
    private Socket socket;
    private BufferedReader in;
    private PrintWriter out;
    private Scanner scanner;

    public WordleClient(String serverAddress, int port) throws IOException {
        socket = new Socket(serverAddress, port);
        in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        out = new PrintWriter(socket.getOutputStream(), true);
        scanner = new Scanner(System.in);
    }

    public void start() {
        try {
            while (true) {
                System.out.println("Enter command: register, login, playWORDLE, sendWord, sendMeStatistics, share, logout");
                String command = scanner.nextLine();
                out.println(command);

                switch (command) {
                    case "register":
                        System.out.println("Enter username:");
                        String username = scanner.nextLine();
                        System.out.println("Enter password:");
                        String password = scanner.nextLine();
                        out.println(username);
                        out.println(password);
                        break;

                    case "login":
                        System.out.println("Enter username:");
                        username = scanner.nextLine();
                        System.out.println("Enter password:");
                        password = scanner.nextLine();
                        out.println(username);
                        out.println(password);
                        break;

                    case "sendWord":
                        System.out.println("Enter guessed word:");
                        String guessedWord = scanner.nextLine();
                        out.println(guessedWord);
                        break;

                    case "logout":
                        System.out.println("Enter username:");
                        username = scanner.nextLine();
                        out.println(username);
                        break;

                    default:
                        System.out.println("Invalid command or unimplemented feature");
                        break;
                }

                String response = in.readLine();
                System.out.println("Server response: " + response);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                socket.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static void main(String[] args) {
        try {
            WordleClient client = new WordleClient("localhost", 12345);
            client.start();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

// WordleServer.java
import java.io.*;
import java.net.*;
import java.util.*;

public class WordleServer {
    private static final int PORT = 12345;
    private static final Map<String, String> users = new HashMap<>(); // username -> password
    private static String secretWord = "APPLE"; // Example secret word
    private static final Set<String> vocabulary = new HashSet<>(Arrays.asList("APPLE", "BERRY", "GRAPE", "MANGO", "PEACH")); // Vocabulary set

    public static void main(String[] args) {
        System.out.println("Wordle Server is running...");
        try (ServerSocket serverSocket = new ServerSocket(PORT)) {
            while (true) {
                new ClientHandler(serverSocket.accept()).start();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static class ClientHandler extends Thread {
        private Socket socket;
        private BufferedReader in;
        private PrintWriter out;
        private String username;

        public ClientHandler(Socket socket) {
            this.socket = socket;
        }

        public void run() {
            try {
                in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                out = new PrintWriter(socket.getOutputStream(), true);

                while (true) {
                    String command = in.readLine();
                    if (command == null) {
                        break;
                    }

                    switch (command) {
                        case "register":
                            handleRegister();
                            break;
                        case "login":
                            handleLogin();
                            break;
                        case "sendWord":
                            handleSendWord();
                            break;
                        case "logout":
                            handleLogout();
                            break;
                        default:
                            out.println("Invalid command");
                            break;
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                try {
                    socket.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        private void handleRegister() throws IOException {
            String username = in.readLine();
            String password = in.readLine();
            if (users.containsKey(username)) {
                out.println("Username already exists");
            } else if (password.isEmpty()) {
                out.println("Password cannot be empty");
            } else {
                users.put(username, password);
                out.println("Registration successful");
            }
        }

        private void handleLogin() throws IOException {
            String username = in.readLine();
            String password = in.readLine();
            if (!users.containsKey(username)) {
                out.println("Username not found");
            } else if (!users.get(username).equals(password)) {
                out.println("Incorrect password");
            } else {
                this.username = username;
                out.println("Login successful");
            }
        }

        private void handleSendWord() throws IOException {
            String guessedWord = in.readLine();
            if (!vocabulary.contains(guessedWord)) {
                out.println("Word not in vocabulary");
            } else {
                StringBuilder feedback = new StringBuilder();
                for (int i = 0; i < secretWord.length(); i++) {
                    char letter = guessedWord.charAt(i);
                    if (secretWord.charAt(i) == letter) {
                        feedback.append('+'); // Correct position
                    } else if (secretWord.indexOf(letter) != -1) {
                        feedback.append('?'); // Wrong position
                    } else {
                        feedback.append('X'); // Not in word
                    }
                }
                out.println("Feedback: " + feedback);
            }
        }

        private void handleLogout() throws IOException {
            out.println("Logout successful for user: " + username);
            this.username = null;
        }
    }
}


//output 2
//server
import java.io.*;
import java.net.*;
import java.util.*;

public class WordleServer {
    private static final int PORT = 12345;
    private static final String SECRET_WORD = "apple"; // Parola segreta
    private static final int MAX_ATTEMPTS = 6;
    
    private static Set<String> registeredUsers = new HashSet<>();
    private static Map<String, Integer> userAttempts = new HashMap<>();

    public static void main(String[] args) throws IOException {
        ServerSocket serverSocket = new ServerSocket(PORT);
        System.out.println("Server avviato sulla porta " + PORT);

        while (true) {
            Socket clientSocket = serverSocket.accept();
            new ClientHandler(clientSocket).start();
        }
    }

    private static class ClientHandler extends Thread {
        private Socket socket;
        private PrintWriter out;
        private BufferedReader in;
        private String username;

        public ClientHandler(Socket socket) {
            this.socket = socket;
        }

        public void run() {
            try {
                out = new PrintWriter(socket.getOutputStream(), true);
                in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                out.println("Benvenuto a Wordle! Registrati con il comando: register [username]");

                while (true) {
                    String input = in.readLine();
                    if (input == null) break;

                    String[] command = input.split(" ");
                    if (command[0].equalsIgnoreCase("register")) {
                        register(command[1]);
                    } else if (command[0].equalsIgnoreCase("play")) {
                        playGame();
                    } else {
                        out.println("Comando non riconosciuto.");
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                try {
                    socket.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        private void register(String username) {
            if (registeredUsers.contains(username)) {
                out.println("Username già registrato.");
            } else {
                this.username = username;
                registeredUsers.add(username);
                userAttempts.put(username, 0);
                out.println("Registrazione completata per " + username + ". Usa 'play' per iniziare.");
            }
        }

        private void playGame() {
            if (userAttempts.get(username) >= MAX_ATTEMPTS) {
                out.println("Hai raggiunto il numero massimo di tentativi.");
                return;
            }

            out.println("Inserisci la tua parola (5 lettere):");
            try {
                String guessedWord = in.readLine();
                if (guessedWord.length() != 5) {
                    out.println("La parola deve essere di 5 lettere.");
                    return;
                }

                userAttempts.put(username, userAttempts.get(username) + 1);
                String feedback = checkGuess(guessedWord);
                out.println(feedback);

                if (guessedWord.equals(SECRET_WORD)) {
                    out.println("Hai indovinato la parola segreta!");
                    userAttempts.put(username, MAX_ATTEMPTS); // Fine del gioco
                } else if (userAttempts.get(username) == MAX_ATTEMPTS) {
                    out.println("Hai esaurito i tentativi. La parola segreta era: " + SECRET_WORD);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        private String checkGuess(String guessedWord) {
            StringBuilder feedback = new StringBuilder();
            for (int i = 0; i < guessedWord.length(); i++) {
                char guessedLetter = guessedWord.charAt(i);
                char secretLetter = SECRET_WORD.charAt(i);
                if (guessedLetter == secretLetter) {
                    feedback.append("+");
                } else if (SECRET_WORD.indexOf(guessedLetter) != -1) {
                    feedback.append("?");
                } else {
                    feedback.append("X");
                }
            }
            return feedback.toString();
        }
    }
}

//client
import java.io.*;
import java.net.*;

public class WordleClient {
    private static final String SERVER_ADDRESS = "localhost";
    private static final int SERVER_PORT = 12345;

    public static void main(String[] args) throws IOException {
        Socket socket = new Socket(SERVER_ADDRESS, SERVER_PORT);
        BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
        BufferedReader userInput = new BufferedReader(new InputStreamReader(System.in));

        String response;
        while ((response = in.readLine()) != null) {
            System.out.println(response);

            if (response.contains("Registrati") || response.contains("Inserisci")) {
                String input = userInput.readLine();
                out.println(input);
            }
        }
    }
}
