import { useTheme } from "../../../context/ThemeContext";
import "./Footer.css";

function Footer() {
  const { isDark } = useTheme();

  return (
    <footer className={`footer ${isDark ? "dark" : ""}`}>
      <div className="footer-container">
        <p className="footer-text">
          Created by <span className="creator-name">Michael Mordehai</span>
        </p>
        <a href="mailto:mordohmichael9@gmail.com" className="footer-email">
          mordohmichael9@gmail.com
        </a>
      </div>
    </footer>
  );
}

export default Footer;
