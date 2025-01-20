// JavaScript file: hello.js

document.addEventListener('DOMContentLoaded', () => {
  const menuToggle = document.querySelector('.toggle');
  const showcase = document.querySelector('.showcase');
  const analyzeLink = document.getElementById('analyze-link');
  const fileInput = document.getElementById('file-input');

  // Toggle menu visibility
  menuToggle.addEventListener('click', () => {
    menuToggle.classList.toggle('active');
    showcase.classList.toggle('active');
  });

  // Trigger file input dialog when the "Analyze your game" link is clicked
  analyzeLink.addEventListener('click', (event) => {
    event.preventDefault(); // Prevent the default action of the link

    fileInput.click(); // Trigger a click on the hidden file input
  });

  // Handle file selection
  fileInput.addEventListener('change', (event) => {
    const files = event.target.files;

    if (files.length > 0) {
      const file = files[0];
      console.log('Selected file:', file);

      // Example of handling the file upload
      // If you need to upload the file, you can use FormData and fetch
      const formData = new FormData();
      formData.append('file', file);

      fetch('/upload-endpoint', { // Replace with your server upload endpoint
        method: 'POST',
        body: formData
      })
      .then(response => response.json())
      .then(data => {
        console.log('Upload successful:', data);
      })
      .catch(error => {
        console.error('Upload error:', error);
      });
    }
  });
});
