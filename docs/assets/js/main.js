
document.addEventListener("DOMContentLoaded", () => {
  // Keep only one audio sample playing at a time.
  const players = Array.from(document.querySelectorAll("audio"));

  players.forEach((player) => {
    player.addEventListener("play", () => {
      players.forEach((other) => {
        if (other !== player) {
          other.pause();
        }
      });
    });
  });

  // Show the expected filename if an audio file is missing.
  players.forEach((player) => {
    player.addEventListener("error", () => {
      const status = player.parentElement?.querySelector(".file-status");

      if (status) {
        status.textContent = `Add file: ${player.dataset.sample}`;
      }
    });
  });
});
