/**
 * Receives recordings from the recording page and saves them in your Google Drive,
 * one folder per person, inside a folder called "Experiment recordings".
 *
 * Set up once (about five minutes):
 *   1. Go to https://script.google.com and click "New project".
 *   2. Delete what is in the editor and paste this whole file.
 *   3. Click Deploy > New deployment. Click the gear and choose "Web app".
 *      Execute as: Me.   Who has access: Anyone.   Click Deploy.
 *   4. Approve the permission prompt (it needs to create files in your Drive).
 *   5. Copy the "Web app URL". That is the address the page sends to.
 *
 * Anyone who has the page can send files here, so it only accepts what the
 * page produces: small video clips with fixed names, and one manifest.
 * To stop receiving, open Deploy > Manage deployments and archive it.
 */

var ROOT_NAME = "Experiment recordings";
var MAX_BYTES = 30 * 1024 * 1024;
var FOLDER_NAME = /^[a-z0-9][a-z0-9-]{0,40}$/;
var TAKE_NAME = /^\d{4}_(aloud|whisper|silent)\.(webm|mp4)$/;

function doPost(e) {
  try {
    var d = JSON.parse(e.postData.contents);
    if (!FOLDER_NAME.test(String(d.folder))) return reply({ ok: false, error: "bad folder name" });
    var folder = personFolder(String(d.folder));

    if (d.kind === "start") return reply({ ok: true });

    if (d.kind === "manifest") {
      var text = String(d.text);
      if (text.length > 2000000) return reply({ ok: false, error: "manifest too large" });
      JSON.parse(text);  // must be real JSON
      replaceFile(folder, Utilities.newBlob(text, "application/json", "manifest.json"));
      return reply({ ok: true });
    }

    if (d.kind === "take") {
      var name = String(d.name);
      if (!TAKE_NAME.test(name)) return reply({ ok: false, error: "bad file name" });
      var bytes = Utilities.base64Decode(String(d.data));
      if (bytes.length === 0 || bytes.length > MAX_BYTES) return reply({ ok: false, error: "bad size" });
      var mime = name.slice(-4) === ".mp4" ? "video/mp4" : "video/webm";
      replaceFile(folder, Utilities.newBlob(bytes, mime, name));
      return reply({ ok: true });
    }

    return reply({ ok: false, error: "unknown request" });
  } catch (err) {
    return reply({ ok: false, error: String(err) });
  }
}

function doGet() {
  return reply({ ok: true, message: "The recording receiver is running." });
}

function reply(body) {
  return ContentService.createTextOutput(JSON.stringify(body)).setMimeType(ContentService.MimeType.JSON);
}

/** The person's folder inside the root folder, created on first use. */
function personFolder(name) {
  var lock = LockService.getScriptLock();
  lock.waitLock(20000);  // several clips arrive at once: only one may create the folder
  try {
    var roots = DriveApp.getFoldersByName(ROOT_NAME);
    var root = roots.hasNext() ? roots.next() : DriveApp.createFolder(ROOT_NAME);
    var found = root.getFoldersByName(name);
    return found.hasNext() ? found.next() : root.createFolder(name);
  } finally {
    lock.releaseLock();
  }
}

/** Save a file, moving any earlier file of the same name to the bin (a redone take). */
function replaceFile(folder, blob) {
  var old = folder.getFilesByName(blob.getName());
  while (old.hasNext()) old.next().setTrashed(true);
  folder.createFile(blob);
}
