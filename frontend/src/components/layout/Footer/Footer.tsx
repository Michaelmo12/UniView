import "./Footer.css";

function Footer() {
  return (
    <footer className="footer">
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
