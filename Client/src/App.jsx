import { useState } from "react";
import "./styles.css";
import useRouter from "./hooks/useRouter";
import Navigation from "./components/Navigation";
import Footer from "./components/Footer";
import HomePage from "./pages/HomePage";
import ChatbotPage from "./pages/ChatbotPage";
import AboutPage from "./pages/AboutPage";
// import HomePage from './pages/HomePage';

const App = () => {
  const [darkMode, setDarkMode] = useState(false);
  const { currentPage, navigate } = useRouter();

  const toggleDarkMode = () => setDarkMode(!darkMode);

  const renderCurrentPage = () => {
    switch (currentPage) {
      case "home":
        return <HomePage navigate={navigate} darkMode={darkMode} />;
      case "chatbot":
        return <ChatbotPage darkMode={darkMode} />;
      case "about":
        return <AboutPage darkMode={darkMode} />;
      default:
        return <HomePage navigate={navigate} darkMode={darkMode} />;
    }
  };
  return (
    <div className={`app-container ${darkMode ? "dark" : ""}`}>
      <Navigation
        currentPage={currentPage}
        navigate={navigate}
        darkMode={darkMode}
        toggleDarkMode={toggleDarkMode}
      />
      <main>{renderCurrentPage()}</main>
      <Footer darkMode={darkMode} />
    </div>
  );
};

export default App;
