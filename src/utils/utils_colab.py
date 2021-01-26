import sys
import subprocess

def is_colab():
    try:
      import google.colab
      return True
    except:
      return False

def install_library_in_colab(librarys):
  """
  :param librarys: [list]
  """

  is_installed = False

  if is_colab():
    # install transformers in colab
    if not is_installed:
      for lib in librarys:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', lib], stdout=subprocess.DEVNULL)
      is_installed = True

  return is_installed



